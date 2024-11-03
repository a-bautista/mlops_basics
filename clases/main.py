#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importar módulos y clases personalizados
from ucimlrepo import fetch_ucirepo
from eda import Diabetes
from eda_plots import Plots
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from models import process_model
from imblearn.over_sampling import RandomOverSampler
import mlflow
from MLFlow import registration_models


# In[2]:


# Ignorar advertencias
warnings.filterwarnings("ignore")


# In[3]:


# Paso 1: Cargar el dataset de indicadores de salud de diabetes
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
modelo = Diabetes(cdc_diabetes_health_indicators)
raw_data = modelo.load_data()

# Paso 2: Exploración y preparación de datos
modelo.explore_data(raw_data)  # Exploración inicial de los datos
modelo.preprocess_conversion_cols(raw_data)  # Preprocesamiento de columnas binarias y categóricas

# Instanciar la clase de visualización
plots = Plots(raw_data, "Diabetes_binary")

# Visualizaciones iniciales
plots.summary_statistics()  # Estadísticas resumidas
plots.plot_numeric_distributions()  # Distribuciones numéricas
plots.plot_binary_counts()  # Conteo de variables binarias
plots.plot_boxplots()  # Diagramas de caja por variable objetivo
plots.plot_crosstab()  # Tablas cruzadas de variables categóricas
plots.plot_correlation_heatmap()  # Heatmap de correlación

# Paso 3: Aplicar transformaciones
temp_data = modelo.explore_different_transformations(raw_data)  # Exploración de transformaciones
transformed_data = modelo.apply_transformations(raw_data)  # Aplicar transformaciones

# Verificar que 'Diabetes_binary' está en el DataFrame transformado
if 'Diabetes_binary' not in transformed_data.columns:
    print("Advertencia: 'Diabetes_binary' no está presente después de las transformaciones.")

# Paso 4: Análisis de componentes principales (PCA)
diabetes_binary_column = transformed_data['Diabetes_binary']  # Guarda la columna objetivo
df_pca, explained_variance = modelo.apply_pca(transformed_data)  # Aplicar PCA
df_pca['Diabetes_binary'] = diabetes_binary_column  # Añadir la columna objetivo de nuevo

# Paso 5: Convertir valores True/False a one-hot encoding
df = modelo.true_false_to_one_hot(df_pca)  # Aplicar one-hot encoding

# Paso 6: Preparar los datos para el modelado
#X = df.drop("Diabetes_binary", axis=1)  # Suponiendo que 'Diabetes_binary' es la variable objetivo
#y = df["Diabetes_binary"]

#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


X_train, X_val, y_train, y_val = modelo.split_data(df)

# Paso 7: Instanciar y evaluar modelos con ProcessModel
process = process_model()
model_evaluator = process.list_models(X_train, y_train, X_val, y_val)


# Paso 8: Registrar el modelo en MLFlow
model_registration = registration_models()


