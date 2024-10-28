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
        data = X.copy()
        data['Diabetes_binary'] = y
        return data

    def explore_data(self, data):
        print(self.filepath.metadata)
        print(self.filepath.variables)
        print(data.head())
        print(data.describe().T)
        print(data.info())
        print(data.shape)
        print(data.isnull().sum())
        print(data.dtypes)

    def preprocess_conversion_cols(self, data):
        binary_columns = [col for col in data.columns if set(data[col].unique()).issubset({0, 1})]
        for col in binary_columns:
            data[col] = data[col].astype('bool')

        categorical_columns = ['GenHlth', 'Age', 'Education', 'Income']
        for col in categorical_columns:
            data[col] = data[col].astype('category')

    def explore_different_transformations(self, data):
        # Create transformed versions of certain columns
        temp_data = data.copy()
        temp_data['Log_BMI'] = np.log(temp_data['BMI'] + 1)
        temp_data['Sqrt_BMI'] = np.sqrt(temp_data['BMI'] + 1)
        pt = PowerTransformer(method='yeo-johnson')
        temp_data[['YeoJohnson_BMI']] = pt.fit_transform(temp_data[['BMI']])

        # Plot the results
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        sns.histplot(data['BMI'], bins=30, kde=True, ax=axes[0, 0])
        sns.histplot(temp_data['Log_BMI'], bins=30, kde=True, ax=axes[1, 0])
        sns.histplot(temp_data['Sqrt_BMI'], bins=30, kde=True, ax=axes[2, 0])
        sns.histplot(temp_data['YeoJohnson_BMI'], bins=30, kde=True, ax=axes[3, 0])
        plt.tight_layout()
        plt.show()

    def apply_transformations(self, data):
        label_encoder = LabelEncoder()
        for col in ['GenHlth', 'Age', 'Education', 'Income']:
            data[col] = label_encoder.fit_transform(data[col])
        
        pt = PowerTransformer(method='yeo-johnson')
        data[['BMI', 'MentHlth', 'PhysHlth']] = pt.fit_transform(data[['BMI', 'MentHlth', 'PhysHlth']])
        
        scaler = StandardScaler()
        data[['BMI', 'MentHlth', 'PhysHlth']] = scaler.fit_transform(data[['BMI', 'MentHlth', 'PhysHlth']])
        return data
    
    def apply_pca(self, data):
        pca = PCA(n_components=7)
        X_pca = pca.fit_transform(data[['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']])
        pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(7)])
        final_df = pd.concat([data.reset_index(drop=True), pca_df], axis=1)
        return final_df

    def true_false_to_one_hot(self, df):
        return df.applymap(lambda x: 1 if x is True else 0)