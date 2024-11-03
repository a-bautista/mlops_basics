import argparse
import pandas as pd
from typing import Text
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import PowerTransformer, StandardScaler
# from sklearn.decomposition import PCA
import yaml

from clases.eda import Diabetes


def featurize(config_path: Text) -> None:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    print('Log: Load dataset')
    dataset = pd.read_csv(config['data']['dataset_csv'])

    # Crea una instancia de Diabetes
    diabetes_instance = Diabetes(dataset)

    # Transform the dataset
    data = diabetes_instance.apply_transformations(dataset)
    diabetes_binary_column = data['Diabetes_binary']  # Guarda la columna objetivo
    df_pca, explained_variance = diabetes_instance.apply_pca(data)

    df_pca['Diabetes_binary'] = diabetes_binary_column  # Añadir la columna objetivo de nuevo

    final_df = diabetes_instance.true_false_to_one_hot(df_pca)  # Aplicar one-hot encoding

    # data = transformation(dataset)
    # final_df = apply_PCA(data)

    print('Log: Save features data file')
    features_path = config['featurize']['features_path']
    final_df.to_csv(features_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)
