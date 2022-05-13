# kaggledays-newyork-3rd-place

- build envirionment

```bash
docker build . -f images/Dockerfile.cuda11 -t kaggle:cuda11
```

- download data to input directory and unzip.

- preprocess

```bash
sh bin/preprocess.sh
```

- make feature

```bash
sh bin/make_feature.sh
```

- train

```bash
sh bin/exp003_lgbm.sh
```

- `./output/exp003_lgbm/sub.csv` is final submission.
