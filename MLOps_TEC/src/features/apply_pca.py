from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

class ApplyPCA:
    def __init__(self, data):
        self.data = data
        self.apply_PCA(data)

    def apply_PCA(self, data):
        X = data[['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']]
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X)

        pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(5)])
        final_df = pd.concat([data.drop(columns=['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']).reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
        print(final_df.head())
        return final_df
