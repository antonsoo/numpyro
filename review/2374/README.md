# Independent NumPyro #2374 checks

Tested source: base `3d990ab703f60e7c37a492becd54eb9a53c22d92`, head `a290139c9083e27ad02d5c6abdad868e1ff9e4df`. CPU, Python 3.14.7, JAX 0.10.2, TFP 0.26.0-dev20261014.

`rq_workload.py` checks spectrum values, length derivatives and second derivatives against independent quadrature of the squared-exponential/Gamma mixture (GPML equation 4.19). It then conditions a spatial GP on 96 DWD temperatures and predicts 64 other stations, using independent dense RQ covariance matrices. Both public HSGP parameterizations are driven with the Gaussian coefficient posterior mean. The one-dimensional controls have identical covariance and posterior arrays on both revisions.

The recorded observations are GSTools' DWD example for June 9, 2020, 12:00. Its 493-row input is pinned by SHA256; 160 deterministic rows are selected, with exact row indices in each report. The coordinates use an explicit local equirectangular projection in units of 300 km. This measures numerical agreement with a specified Euclidean GP, not geographic-model suitability or forecast accuracy. Kernel and noise parameters are fixed, with no inference over scale_mixture (whose order derivative TFP does not provide).

Data provenance: https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/08_geo_coordinates/01_dwd_krige.html
Mathematical source: https://gaussianprocess.org/gpml/chapters/RW4.pdf

Run from a NumPyro checkout with the optional TFP dependency:

```sh
curl -L https://raw.githubusercontent.com/GeoStat-Framework/GSTools/47297c4aac5c3b43d8f593e6764bc9f3da600f3d/examples/08_geo_coordinates/temp_obs.txt -o /tmp/dwd-temp.txt
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python review/2374/rq_workload.py --source . --data /tmp/dwd-temp.txt --output /tmp/rq-head.json
# Add --float32 to repeat at the default precision.
```

Use the same script with `--source` pointing to each source tree. It explicitly pins and asserts the import. Original reports are included, with local filesystem paths removed. The full head HSGP suite passes 161 tests in x64; the TFP/JAX deprecation warning remains. On the base, 10 of the author's 18 new normalization cases fail numerically (8 pass). No full-repository, accelerator, hyperparameter-fitting or general approximation-accuracy claim is made.
