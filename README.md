# mlflow_tutorial

## mlflow ui 실행
```bash
mlflow ui --port=5000 &
```


## train.py 파일로 바로 실행
```bash
python3 ./mnist/train.py {epochs} {batch_size}
```


## Command Line으로 MLproject 실행
```bash
mlflow run --no-conda mnist -P epochs={epochs} -P batch_size={batch_size}
```


## mlflow Python API로 MLproject 실행
```bash
python3 ./mnist/parameter_tunning.py
```