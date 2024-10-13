class function_fase_1:
    def __init__(self):
        pass
    class load_data:
        def __init__(self,filepath):
            self.data, self.filepath=self.load_data(filepath)
        def load_data(self,filepath):
        # data (as pandas dataframes)
            X = filepath.data.features
            y = filepath.data.targets
            # Merge them into a single DataFrame
            data = X.copy()
            data['Diabetes_binary'] = y
            return data,filepath
    
    class explore_data:
        def __init__(self,data,filepath):
            self.data=data
            self.filepath=filepath
            self.explore_data(data,filepath)
    
        def explore_data(self,data, filepath):
            print(filepath.metadata)  ## considerar
            print(filepath.variables) ## considerar
            print(data.head())
            print(data.describe().T) ## considerar
            print(data.info()) ## considerar
            print(data.shape)
            print(data.isnull().sum())
            
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
            print(data.dtypes)
        
    class plot_graph:
    # 1. Summary Statistics
        def __init__(self,data):
            self.data=data
            self.summary_statistics(data)
            self.plot_numeric_distributions(data)
            self.plot_binary_counts(data,'Diabetes_binary')
            self.plot_boxplots(data,'Diabetes_binary')
            self.plot_crosstab(data,'Diabetes_binary')
            self.plot_correlation_heatmap(data)
        
        def summary_statistics(self,data):
            print("Summary Statistics:")
            print(data.describe(include='all'))
        
        # 2. Distribution of Numeric Variables
        def plot_numeric_distributions(self,data):
            numeric_columns = data.select_dtypes(include=['int64']).columns
            for col in numeric_columns:
              plt.figure(figsize=(10, 5))
              sns.histplot(data[col], bins=30, kde=True)
              plt.title(f'Distribution of {col}')
              plt.xlabel(col)
              plt.ylabel('Frequency')
              plt.grid()
              plt.show()
        
        # 3. Count Plots for Binary Variables against the Target
        def plot_binary_counts(self,data, target):
            binary_columns = data.select_dtypes(include=['bool']).columns
            for col in binary_columns:
              plt.figure(figsize=(10, 5))
              sns.countplot(x=data[col], hue=data[target])
              plt.title(f'Count of {col} by {target}')
              plt.xlabel(col)
              plt.ylabel('Count')
              plt.legend(title=target)
              plt.grid()
              plt.show()
            
        # 4. Box Plots for Continuous Variables by Target Variable
        def plot_boxplots(self,data, target):
            numeric_columns = data.select_dtypes(include=['int64']).columns
            for num_col in numeric_columns:
              plt.figure(figsize=(10, 5))
              sns.boxplot(x=data[target], y=data[num_col])
              plt.title(f'Boxplot of {num_col} by {target}')
              plt.xlabel(target)
              plt.ylabel(num_col)
              plt.grid()
              plt.show()
        
        # 5. Crosstabulation for Categorical Variables against the Target
        def plot_crosstab(self,data, target):
            categorical_columns = data.select_dtypes(include=['category']).columns
            for cat_col in categorical_columns:
              crosstab = pd.crosstab(data[cat_col], data[target])
              print(f'Crosstab for {cat_col} vs {target}:')
              print(crosstab)
              sns.heatmap(crosstab, annot=True, fmt="d", cmap='Blues')
              plt.title(f'Crosstab Heatmap of {cat_col} by {target}')
              plt.ylabel(cat_col)
              plt.xlabel(target)
              plt.show()
        
        # 6. Correlation Heatmap for Numeric Variables
        def plot_correlation_heatmap(self,data):
            numeric_data = data.select_dtypes(include=['int64'])
            correlation_matrix = numeric_data.corr()
            plt.figure(figsize=(15, 15))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
            plt.title('Correlation Heatmap')
            plt.show()
    
    class transformation_type:
        def __init__(self,data):
            self.data=data
            self.transformation_type(data)
        
        def transformation_type(self,data):
        # Create a temporal DataFrame  for the transformations
            temp_data = data.copy()
            
            # Create the transformations in the temporary DataFrame
            temp_data['Log_BMI'] = np.log(temp_data['BMI'] + 1)
            temp_data['Log_MentHlth'] = np.log(temp_data['MentHlth'] + 1)
            temp_data['Log_PhysHlth'] = np.log(temp_data['PhysHlth'] + 1)
            
            temp_data['Sqrt_BMI'] = np.sqrt(temp_data['BMI'] + 1)
            temp_data['Sqrt_MentHlth'] = np.sqrt(temp_data['MentHlth'] + 1)
            temp_data['Sqrt_PhysHlth'] = np.sqrt(temp_data['PhysHlth'] + 1)
            
            # Yeo-Johnson Transformation
            pt = PowerTransformer(method='yeo-johnson')
            temp_data[['YeoJohnson_BMI', 'YeoJohnson_MentHlth', 'YeoJohnson_PhysHlth']] = pt.fit_transform(temp_data[['BMI', 'MentHlth', 'PhysHlth']])
            
            # Configure the figure for multiple subgraphs
            fig, axes = plt.subplots(5, 3, figsize=(15, 20))
            fig.suptitle('Comparison of Distributions of Numerical Variables and Transforms', fontsize=16)
            
            # Originals
            for i, col in enumerate(['BMI', 'MentHlth', 'PhysHlth']):
              sns.histplot(data[col], bins=30, kde=True, ax=axes[0, i])
              axes[0, i].set_title(f'Distribución de {col}')
              axes[0, i].set_xlabel(col)
              axes[0, i].set_ylabel('Frecuencia')
            
            # Logarithmic Transformation
            for i, col in enumerate(['Log_BMI', 'Log_MentHlth', 'Log_PhysHlth']):
              sns.histplot(temp_data[col], bins=30, kde=True, ax=axes[1, i])
              axes[1, i].set_title(f'Distribución Log de {col}')
              axes[1, i].set_xlabel(col)
              axes[1, i].set_ylabel('Frecuencia')
            
            # Square Root Transformation
            for i, col in enumerate(['Sqrt_BMI', 'Sqrt_MentHlth', 'Sqrt_PhysHlth']):
              sns.histplot(temp_data[col], bins=30, kde=True, ax=axes[2, i])
              axes[2, i].set_title(f'Distribución Raíz Cuadrada de {col}')
              axes[2, i].set_xlabel(col)
              axes[2, i].set_ylabel('Frecuencia')
            
            # Yeo-Johnson Transformation
            for i, col in enumerate(['YeoJohnson_BMI', 'YeoJohnson_MentHlth', 'YeoJohnson_PhysHlth']):
              sns.histplot(temp_data[col], bins=30, kde=True, ax=axes[3, i])
              axes[3, i].set_title(f'Distribución Yeo-Johnson de {col}')
              axes[3, i].set_xlabel(col)
              axes[3, i].set_ylabel('Frecuencia')
            
            # Occupying the space in the last row (you can adjust or add more graphs)
            for ax in axes[4]:
              ax.axis('off')  #Or you can add other graphs here
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  #  Adjust design
            plt.show()
        
    class transformation:
        def __init__(self,data):
            self.data=data
            self.codification(data)
            self.transform(data)
        # 1. Codification
        def codification(self,data):
        # Inicializa el codificador
            label_encoder = LabelEncoder()
            
            # Aplica el Label Encoding a las columnas ordinales
            data['GenHlth'] = label_encoder.fit_transform(data['GenHlth'])
            data['Age'] = label_encoder.fit_transform(data['Age'])
            data['Education'] = label_encoder.fit_transform(data['Education'])
            data['Income'] = label_encoder.fit_transform(data['Income'])
            
            # Muestra el DataFrame transformado
            print(data[['GenHlth', 'Age', 'Education','Income']])
            return data
        
        # 2. Transformation
        def transform(self,data):
        # Aplicar Yeo-Johnson y reemplazar las columnas originales
            pt = PowerTransformer(method='yeo-johnson')
            data[['BMI', 'MentHlth', 'PhysHlth']] = pt.fit_transform(data[['BMI', 'MentHlth', 'PhysHlth']])
            
            # Estandarizar las variables transformadas y reemplazar las columnas originales
            scaler = StandardScaler()
            data[['BMI', 'MentHlth', 'PhysHlth']] = scaler.fit_transform(data[['BMI', 'MentHlth', 'PhysHlth']])
            
            # Mostrar el DataFrame transformado y estandarizado
            print(data[['BMI', 'MentHlth', 'PhysHlth']])
            return data
    
    class process_apply_PCA:
        def __init__(self,data):
            self.data=data
            self.final_df=self.apply_PCA(data)
            
        def apply_PCA(self,data):

        # Seleccionar características para PCA (variables ya transformadas y escaladas)
            X = data[['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']]
            
            # Inicializa PCA y ajusta el modelo
            pca = PCA()
            pca.fit(X)
            
            # Variancia explicada por cada componente
            explained_variance = pca.explained_variance_ratio_
            
            # Crea un gráfico de codo
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
            plt.title('Varianza Explicada por Componentes Principales')
            plt.xlabel('Número de Componentes Principales')
            plt.ylabel('Proporción de Varianza Explicada')
            plt.grid()
            plt.show()
            
            # Crea un gráfico de varianza acumulada
            plt.figure(figsize=(10, 6))
            cumulative_variance = explained_variance.cumsum()
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
            plt.title('Varianza Acumulada por Componentes Principales')
            plt.xlabel('Número de Componentes Principales')
            plt.ylabel('Varianza Acumulada')
            plt.grid()
            plt.axhline(y=0.92, color='r', linestyle='--')  # Umbral del 92%
            plt.show()
            
            # Seleccionar características para PCA (variables ya transformadas y escaladas)
            X = data[['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']]
            
            # Aplicar PCA
            pca = PCA(n_components=5)  # Elegir cuántas componentes principales mantener
            X_pca = pca.fit_transform(X)
            
            # Convertir el resultado a un DataFrame
            pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(5)])
            
            # Unir las componentes principales con el DataFrame original, excluyendo las variables originales
            final_df = pd.concat([data.drop(columns=['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age', 'Education', 'Income']).reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
            
            # Mostrar el DataFrame final
            print(final_df.head())
            return final_df
            
                        
                        # Run EDA Functions