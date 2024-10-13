import argparse
import pandas as pd
from typing import Text
import yaml
from sklearn.model_selection import train_test_split

def data_split(config_path: Text) -> None:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    print('Log: Load features')
    dataset = pd.read_csv(config['featurize']['features_path'])
    
    # return train_test_split(df.drop(target, axis='columns'), df[target], train_size=0.80, random_state=10,stratify=df[target])

    X = dataset.drop('Diabetes_binary', axis='columns')
    y = dataset['Diabetes_binary']

    print('Log: Split features into X_train, X_test, y_train, y_test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['data_split']['test_size'], random_state=config['base']['random_state'])

    print('Log: Save train and test files')
    train_csv_path = config['data_split']['trainset_path']
    test_csv_path = config['data_split']['testset_path']
    train_target_csv_path = config['data_split']['train_target']
    test_target_csv_path = config['data_split']['test_target']

    X_train.to_csv(train_csv_path, index=False)
    X_test.to_csv(test_csv_path, index=False)
    y_train.to_csv(train_target_csv_path, index=False)
    y_test.to_csv(test_target_csv_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)