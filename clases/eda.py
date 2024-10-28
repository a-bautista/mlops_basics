import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.decomposition import PCA

class Diabetes:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        X = self.filepath.data.features
        y = self.filepath.data.targets
        # get the dataset with all the variables
        data = X.copy()
        data['Diabetes_binary'] = y
        return data

    def explore_data(self, data):
        print(self.filepath.metadata)  ## considerar
        print(self.filepath.variables) ## considerar
        print(data.head())
        print(data.describe().T) ## considerar
        print(data.info()) ## considerar
        print(data.shape)
        print(data.isnull().sum())
        print(data.dtypes)
        
    def preprocess_conversion_cols(self, data):
        # Automatically find binary columns
        binary_columns = [col for col in data.columns if set(data[col].unique()).issubset({0, 1})]
        
        # Convert found binary columns to bool
        for col in binary_columns:
            data[col] = data[col].astype('bool')
        
        # List of categorical columns to convert
        categorical_columns = ['GenHlth', 'Age', 'Education', 'Income']
        
        # Convert specified columns to category
        for col in categorical_columns:
            data[col] = data[col].astype('category')
        
        # Check the data types
        #print(data.dtypes)

    def explore_different_transformations(self, data):
        # Crear un DataFrame temporal para las transformaciones
        temp_data = data.copy()

        # Crear las transformaciones en el DataFrame temporal
        temp_data['Log_BMI'] = np.log(temp_data['BMI'] + 1)
        temp_data['Log_MentHlth'] = np.log(temp_data['MentHlth'] + 1)
        temp_data['Log_PhysHlth'] = np.log(temp_data['PhysHlth'] + 1)

        temp_data['Sqrt_BMI'] = np.sqrt(temp_data['BMI'] + 1)
        temp_data['Sqrt_MentHlth'] = np.sqrt(temp_data['MentHlth'] + 1)
        temp_data['Sqrt_PhysHlth'] = np.sqrt(temp_data['PhysHlth'] + 1)

        # Yeo-Johnson Transformation
        pt = PowerTransformer(method='yeo-johnson')
        temp_data[['YeoJohnson_BMI', 'YeoJohnson_MentHlth', 'YeoJohnson_PhysHlth']] = pt.fit_transform(temp_data[['BMI', 'MentHlth', 'PhysHlth']])
        return temp_data 


    def apply_transformations(self, data):

        label_encoder = LabelEncoder()    
        # Aplica el Label Encoding a las columnas ordinales
        data['GenHlth'] = label_encoder.fit_transform(data['GenHlth'])
        data['Age'] = label_encoder.fit_transform(data['Age'])
        data['Education'] = label_encoder.fit_transform(data['Education'])
        data['Income'] = label_encoder.fit_transform(data['Income'])
        
        # Muestra el DataFrame transformado
        print(data[['GenHlth', 'Age', 'Education','Income']])
        pt = PowerTransformer(method='yeo-johnson')
        data[['BMI', 'MentHlth', 'PhysHlth']] = pt.fit_transform(data[['BMI', 'MentHlth', 'PhysHlth']])
        
        # Estandarizar las variables transformadas y reemplazar las columnas originales
        scaler = StandardScaler()
        data[['BMI', 'MentHlth', 'PhysHlth']] = scaler.fit_transform(data[['BMI', 'MentHlth', 'PhysHlth']])
        
        # Mostrar el DataFrame transformado y estandarizado
        print(data[['BMI', 'MentHlth', 'PhysHlth']])
        return data
    
    def apply_pca(self, data):

        X = data[['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']]

        # Inicializa PCA y ajusta el modelo
        # Elegir cuántas componentes principales mantener
        pca = PCA() 
        X_pca = pca.fit_transform(X)

        # Variancia explicada por cada componente
        explained_variance = pca.explained_variance_ratio_

        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        return pca_df, explained_variance  # Retorna el DataFrame de PCA y la varianza explicada


    def true_false_to_one_hot(self,df):
        df_converted=df.applymap(lambda x: 1 if x is True else (0 if x is False else x))
        return df_converted