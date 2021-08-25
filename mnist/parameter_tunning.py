import mlflow

project_uri = "/Users/yeomyungro/Documents/github/mlflow_tutorial/mnist"
params = {"epochs": 15, "batch_size": 128}

mlflow.run(project_uri, parameters=params, use_conda=False)