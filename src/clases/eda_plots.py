import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from math import ceil


class Plots:
    # 1. Summary Statistics
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def confusion_matrix(self, labels):
        plt.figure(figsize=(6, 4))
        ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False)
        ax.set(ylabel="Real labels", xlabel="Prediction labels")
        plt.show()

    def plot_distributions_and_pca(self, data, temp_data):
        # Configurar la figura para múltiples subgráficas
        fig, axes = plt.subplots(5, 3, figsize=(15, 20))
        fig.suptitle('Comparación de Distribuciones de Variables Numéricas y Transformaciones', fontsize=16)

        # Originales
        for i, col in enumerate(['BMI', 'MentHlth', 'PhysHlth']):
            sns.histplot(data[col], bins=30, kde=True, ax=axes[0, i])
            axes[0, i].set_title(f'Distribución de {col}')
            axes[0, i].set_xlabel(col)
            axes[0, i].set_ylabel('Frecuencia')

        # Transformación Logarítmica
        for i, col in enumerate(['Log_BMI', 'Log_MentHlth', 'Log_PhysHlth']):
            sns.histplot(temp_data[col], bins=30, kde=True, ax=axes[1, i])
            axes[1, i].set_title(f'Distribución Log de {col}')
            axes[1, i].set_xlabel(col)
            axes[1, i].set_ylabel('Frecuencia')

        # Transformación de Raíz Cuadrada
        for i, col in enumerate(['Sqrt_BMI', 'Sqrt_MentHlth', 'Sqrt_PhysHlth']):
            sns.histplot(temp_data[col], bins=30, kde=True, ax=axes[2, i])
            axes[2, i].set_title(f'Distribución Raíz Cuadrada de {col}')
            axes[2, i].set_xlabel(col)
            axes[2, i].set_ylabel('Frecuencia')

        # Transformación Yeo-Johnson
        for i, col in enumerate(['YeoJohnson_BMI', 'YeoJohnson_MentHlth', 'YeoJohnson_PhysHlth']):
            sns.histplot(temp_data[col], bins=30, kde=True, ax=axes[3, i])
            axes[3, i].set_title(f'Distribución Yeo-Johnson de {col}')
            axes[3, i].set_xlabel(col)
            axes[3, i].set_ylabel('Frecuencia')

        # Ocupando el espacio en la última fila
        for ax in axes[4]:
            ax.axis('off')  # Espacio para otras gráficas si es necesario

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustar layout
        plt.show()

        # Variancia explicada por cada componente principal (PCA)
        X = data[['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']]
        pca = PCA()
        X_pca = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_

        # Gráfico de codo
        plt.figure(figsize=(6, 6))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
        plt.title('Varianza Explicada por Componentes Principales')
        plt.xlabel('Número de Componentes Principales')
        plt.ylabel('Proporción de Varianza Explicada')
        plt.grid()
        plt.show()

        # Gráfico de varianza acumulada
        cumulative_variance = explained_variance.cumsum()
        plt.figure(figsize=(6, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        plt.title('Varianza Acumulada por Componentes Principales')
        plt.xlabel('Número de Componentes Principales')
        plt.ylabel('Varianza Acumulada')
        plt.grid()
        plt.axhline(y=0.92, color='r', linestyle='--')  # Umbral del 92%
        plt.show()

        num_components = np.where(cumulative_variance >= 0.99)[0][0] + 1
        print(f"El número mínimo de componentes principales que explica más del 99% de la varianza es: {num_components}")

        # PCA con n_componentes seleccionados
        pca = PCA(n_components=7)
        X_pca = pca.fit_transform(X)

        # Convertir a DataFrame y unir con el original
        pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(7)])
        final_df = pd.concat([data.drop(columns=['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']).reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)

        print(final_df.head())
        return final_df

    def summary_statistics(self):
        print("Summary Statistics:")
        print(self.data.describe(include='all'))

    # 2. Distribution of Numeric Variables
    def plot_numeric_distributions(self):
        numeric_columns = self.data.select_dtypes(include=['int64']).columns
        n_cols = 3  # Number of columns in the grid
        n_rows = ceil(len(numeric_columns) / n_cols)  # Calculate rows needed

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, col in enumerate(numeric_columns):
            sns.histplot(self.data[col], bins=30, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True)

        # Hide any extra subplots that are not used
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    # 3. Count Plots for Binary Variables against the Target
    def plot_binary_counts(self):
        binary_columns = self.data.select_dtypes(include=['bool']).columns
        n_cols = 3  # Number of columns in the grid
        n_rows = ceil(len(binary_columns) / n_cols)  # Calculate rows needed

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, col in enumerate(binary_columns):
            sns.countplot(x=self.data[col], hue=self.data[self.target], ax=axes[i])
            axes[i].set_title(f'Count of {col} by {self.target}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
            axes[i].legend(title=self.target)
            axes[i].grid(True)

        # Hide any extra subplots that are not used
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    # 4. Box Plots for Continuous Variables by Target Variable
    def plot_boxplots(self):
        numeric_columns = self.data.select_dtypes(include=['int64']).columns
        n_cols = 3  # Number of columns in the grid
        n_rows = ceil(len(numeric_columns) / n_cols)  # Calculate rows needed

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, num_col in enumerate(numeric_columns):
            sns.boxplot(x=self.data[self.target], y=self.data[num_col], ax=axes[i])
            axes[i].set_title(f'Boxplot of {num_col} by {self.target}')
            axes[i].set_xlabel(self.target)
            axes[i].set_ylabel(num_col)
            axes[i].grid(True)

        # Hide any extra subplots that are not used
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    # 5. Crosstabulation for Categorical Variables against the Target
    def plot_crosstab(self):
        categorical_columns = self.data.select_dtypes(include=['category']).columns
        for cat_col in categorical_columns:
            crosstab = pd.crosstab(self.data[cat_col], self.data[self.target])
            print(f'Crosstab for {cat_col} vs {self.target}:')
            print(crosstab)
            sns.heatmap(crosstab, annot=True, fmt="d", cmap='Blues')
            plt.title(f'Crosstab Heatmap of {cat_col} by {self.target}')
            plt.ylabel(cat_col)
            plt.xlabel(self.target)
            plt.show()

    # 6. Correlation Heatmap for Numeric Variables
    def plot_correlation_heatmap(self):
        numeric_data = self.data.select_dtypes(include=['int64'])
        correlation_matrix = numeric_data.corr()
        plt.figure(figsize=(8, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Heatmap')
        plt.show()
