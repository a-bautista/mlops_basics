import argparse
import joblib
import pandas as pd
from typing import Text
import yaml
from sklearn.linear_model import LogisticRegression

def train_model(config_path: Text) -> None:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    print('Log: Load train dataset')
    X_train = pd.read_csv(config['data_split']['trainset_path'])
    y_train = pd.read_csv(config['data_split']['train_target'])


    print('Log: Train model')
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train.values.ravel())

    print('Log: Save model')
    models_path = config['train']['model_path']
    joblib.dump(model, models_path)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)