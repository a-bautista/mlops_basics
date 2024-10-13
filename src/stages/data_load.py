import argparse
from ucimlrepo import fetch_ucirepo
import pandas as pd
from typing import Text
import yaml

def data_load(config_path: Text) -> None:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    #config = yaml.safe_load(open(config_path))
    #raw_data_path = config['data_load']['raw_data_path']

    print('Log: Data load')
    # fetch dataset
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

    # data (as pandas dataframes)
    X = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets

    # Merge them into a single DataFrame
    data = X.copy()
    data['Diabetes_binary'] = y

    print('Log: Save raw data file')
    data.to_csv(config['data']['dataset_csv'], index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
