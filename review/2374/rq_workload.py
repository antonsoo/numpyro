# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Independent RQ spectrum derivatives and recorded spatial GP conditioning.

Run each revision in a fresh process. The input file is read only as text.
No GSTools or downloaded data code is imported or executed.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.special import gamma

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--data", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--float32", action="store_true")
a = p.parse_args()
source = a.source.resolve()
sys.path.insert(0, str(source))
import jax
import jax.numpy as jnp
import numpyro
from numpyro.contrib.hsgp.approximation import hsgp_rational_quadratic
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_rational_quadratic,
    spectral_density_rational_quadratic,
)
from numpyro.handlers import seed, substitute

assert Path(numpyro.__file__).resolve().parent.parent == source
jax.config.update("jax_enable_x64", not a.float32)
jax.config.update("jax_default_matmul_precision", "highest")
print("SOURCE", numpyro.__file__, "PYTHON", sys.executable, "JAX", jax.__version__, flush=True)
spectra = []


def reference(dim, frequency, amplitude, length, mixture):
    # GPML eq.4.19: integrate SE spectra against the defining Gamma mixture.
    # Differentiate that integral analytically, independently of Bessel K.
    c = amplitude * (2 * np.pi * mixture * length**2) ** (dim / 2) / gamma(mixture)

    def integral(order):
        def integrand(t):
            q = mixture * length * frequency**2 / t
            log_derivative = dim / length - q
            factor = 1.0 if order == 0 else log_derivative
            if order == 2:
                factor = log_derivative**2 - dim / length**2 - q / length
            return t ** (mixture - 1 - dim / 2) * np.exp(
                -t - mixture * length**2 * frequency**2 / (2 * t)
            ) * factor

        return c * quad(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-10, limit=200)[0]

    return np.array([integral(i) for i in range(3)])


for dim, length, frequency in itertools.product([1, 2, 3], [.4, 1., 1.8], [0., .3, 1., 3.]):
    w = np.zeros(dim)
    w[0] = frequency

    def f(l):
        return spectral_density_rational_quadratic(dim, w, 1.7, l, 3.)

    actual = np.array([f(length), jax.grad(f)(length), jax.grad(jax.grad(f))(length)])
    expected = reference(dim, frequency, 1.7, length, 3.)
    spectra.append(dict(dim=dim, length=length, frequency=frequency,
                        actual=actual.tolist(), reference=expected.tolist(),
                        scaled_error=(np.abs(actual - expected) / np.maximum(1., np.abs(expected))).tolist()))
print("SPECTRA", len(spectra), "max errors", np.max([r["scaled_error"] for r in spectra], axis=0), flush=True)

assert hashlib.sha256(a.data.read_bytes()).hexdigest() == "7f5d21e4b9cd1dd192f9d44119a10dafdd52e7a11c1f6fa7c521ecb76746c179"
data = np.loadtxt(a.data)
rows = np.random.default_rng(2374).permutation(len(data))[:160]
d = data[rows]
# Explicit local equirectangular coordinates in units of 300 km. This tests
# implementation of an Euclidean GP, not the suitability of a geographic model.
lat0, lon0 = d[:, 1:3].mean(axis=0)
x = np.column_stack(((d[:, 2] - lon0) * 111.195 * np.cos(np.deg2rad(lat0)),
                     (d[:, 1] - lat0) * 111.195)) / 300.
y = (d[:, 3] - d[:96, 3].mean()) / d[:96, 3].std()
ntrain = 96
amplitude, mixture, noise = 1.7, 3., .2
records, arrays = [], {}


def condition(K):
    train = K[:ntrain, :ntrain] + noise**2 * np.eye(ntrain)
    factor = cho_factor(train, lower=True)
    dual = cho_solve(factor, y[:ntrain])
    cross = K[ntrain:, :ntrain]
    mean = cross @ dual
    cov = K[ntrain:, ntrain:] - cross @ cho_solve(factor, cross.T)
    nll = .5 * y[:ntrain] @ dual + np.log(np.diag(factor[0])).sum() + .5 * ntrain * np.log(2*np.pi)
    return mean, cov, nll, dual


for dim in [1, 2]:
    points = x[:, :dim]
    distance2 = cdist(points, points, metric="sqeuclidean")
    for length in [.4, 1., 1.8]:
        ell = float(np.max(np.abs(points)) + 10 * length)
        m = [96] * dim
        phi = np.asarray(eigenfunctions(jnp.asarray(points), ell=ell, m=m))
        # Independent NumPy sine-product basis, including Cartesian ordering.
        indices = np.array(list(itertools.product(range(1, 97), repeat=dim)))
        omega = indices * np.pi / (2 * ell)
        phi_reference = np.prod(np.sin((points[:, None, :] + ell) * omega[None, :, :]) / np.sqrt(ell), axis=-1)
        np.testing.assert_allclose(phi, phi_reference,
                                   rtol=1e-4 if a.float32 else 1e-11,
                                   atol=512*np.finfo(np.float32).eps if a.float32 else 1e-13)
        spd = np.asarray(diag_spectral_density_rational_quadratic(
            amplitude, length, mixture, ell, m, dim))
        approx = (phi * spd) @ phi.T
        exact = amplitude * (1 + distance2 / (2 * mixture * length**2)) ** -mixture
        mean, cov, nll, dual = condition(approx)
        refmean, refcov, refnll, _ = condition(exact)
        # Drive the public probabilistic model with the actual Gaussian
        # coefficient posterior mean, for both parameterizations.
        coefficients = spd * (phi[:ntrain].T @ dual)
        public_error = []
        for noncentered in [False, True]:
            beta = coefficients / np.sqrt(spd) if noncentered else coefficients
            call = seed(substitute(hsgp_rational_quadratic, data={"beta": jnp.asarray(beta)}), 0)
            actual = call(jnp.asarray(points), amplitude, length, mixture, ell, m, noncentered)
            public_error.append(float(np.max(np.abs(np.asarray(actual)[ntrain:] - mean))))
        rec = dict(dim=dim, length=length, ell=ell, m=m, basis_count=phi.shape[-1],
                   basis_reference_error=float(np.max(np.abs(phi-phi_reference))),
                   max_covariance_error=float(np.max(np.abs(approx-exact))),
                   mean_prior_variance=float(np.diag(approx).mean()),
                   max_prediction_error=float(np.max(np.abs(mean-refmean))),
                   max_posterior_variance_error=float(np.max(np.abs(np.diag(cov)-np.diag(refcov)))),
                   nll_error=float(abs(nll-refnll)), public_prediction_errors=public_error,
                   condition_number=float(np.linalg.cond(exact[:ntrain,:ntrain]+noise**2*np.eye(ntrain))),
                   nll=float(nll), reference_nll=float(refnll))
        key = f"d{dim}_l{length}"
        for name, value in [("covariance",approx),("posterior_mean",mean),("posterior_covariance",cov),
                            ("reference_mean",refmean),("reference_covariance",refcov)]:
            arrays[key+"_"+name] = value
        records.append(rec)
        print("GP", rec, flush=True)

result = dict(source=str(source), imported=numpyro.__file__, python=sys.version,
              executable=sys.executable, jax=jax.__version__, numpy=np.__version__,
              float32=a.float32,
              commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip(),
              dirty=subprocess.check_output(["git","status","--porcelain"],cwd=source,text=True),
              data_hash=hashlib.sha256(a.data.read_bytes()).hexdigest(), data_rows=len(data),
              selected_rows=rows.tolist(), train=ntrain, held_out=len(y)-ntrain,
              coordinate_unit_km=300, amplitude=amplitude, mixture=mixture, noise_sd=noise,
              spectrum_checks=spectra, gp_checks=records)
a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
np.savez_compressed(a.output.with_suffix(".npz"), **arrays)
