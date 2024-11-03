import argparse
from ucimlrepo import fetch_ucirepo
from typing import Text
import yaml

from clases.eda import Diabetes

def data_load(config_path: Text) -> None:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    print('Log: Data load') 
    # fetch dataset
    #cdc_diabetes_health_indicators = fetch_ucirepo(id=config['data']['ucirepo_id'])
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

    diabetes_instance = Diabetes(cdc_diabetes_health_indicators)
    raw_data = diabetes_instance.load_data()

    print('Log: Save raw data file')
    raw_data.to_csv(config['data']['dataset_csv'], index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
