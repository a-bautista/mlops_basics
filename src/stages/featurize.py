import argparse
import pandas as pd
from typing import Text
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
import yaml

#Function to apply transformation
def transformation(data):

  # 1. Codification
  def codification(data):
    # Inicializa el codificador
    label_encoder = LabelEncoder()

    # Aplica el Label Encoding a las columnas ordinales
    data['GenHlth'] = label_encoder.fit_transform(data['GenHlth'])
    data['Age'] = label_encoder.fit_transform(data['Age'])
    data['Education'] = label_encoder.fit_transform(data['Education'])
    data['Income'] = label_encoder.fit_transform(data['Income'])

    return data

  # 2. Transformation
  def transform(data):
    # Aplicar Yeo-Johnson y reemplazar las columnas originales
    pt = PowerTransformer(method='yeo-johnson')
    data[['BMI', 'MentHlth', 'PhysHlth']] = pt.fit_transform(data[['BMI', 'MentHlth', 'PhysHlth']])

    # Estandarizar las variables transformadas y reemplazar las columnas originales
    scaler = StandardScaler()
    data[['BMI', 'MentHlth', 'PhysHlth']] = scaler.fit_transform(data[['BMI', 'MentHlth', 'PhysHlth']])

    # Mostrar el DataFrame transformado y estandarizado
    print(data[['BMI', 'MentHlth', 'PhysHlth']])
    return data

  return data


  # Run transformation Functions
  codification(data)
  transform(data)


  #Function to apply PCA
def apply_PCA(data):

  # Seleccionar características para PCA (variables ya transformadas y escaladas)
  X = data[['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']]

  # Inicializa PCA y ajusta el modelo
  pca = PCA()
  pca.fit(X)

  # Seleccionar características para PCA (variables ya transformadas y escaladas)
  X = data[['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']]

  # Aplicar PCA
  pca = PCA(n_components=5)  # Elegir cuántas componentes principales mantener
  X_pca = pca.fit_transform(X)

  # Convertir el resultado a un DataFrame
  pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(5)])

  # Unir las componentes principales con el DataFrame original, excluyendo las variables originales
  final_df = pd.concat([data.drop(columns=['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']).reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)

  return final_df


def featurize(config_path: Text) -> None:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    print('Log: Load dataset')
    dataset = pd.read_csv(config['data']['dataset_csv'])
    
    # Transform the dataset
    data = transformation(dataset)
    final_df = apply_PCA(data) 

    print('Log: Save features data file')
    features_path = config['featurize']['features_path']
    final_df.to_csv(features_path, index=False)

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)