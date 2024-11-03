# Importar módulos y clases personalizados
from ucimlrepo import fetch_ucirepo
# from data_preparation import DataPreparation
from eda import Diabetes
from eda_plots import Plots
import warnings

warnings.filterwarnings("ignore")
# Inicializar instancias de clases# Paso 1: Cargar el dataset de indicadores de salud de diabetes
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
modelo = Diabetes(cdc_diabetes_health_indicators)
raw_data = modelo.load_data()

# Paso 2: Exploración y preparación de datos
modelo.explore_data(raw_data)  # Exploración inicial de los datos
modelo.preprocess_conversion_cols(raw_data)  # Preprocesamiento de columnas binarias y categóricas

# Instanciar clases adicionales para el flujo de EDA y Plots
eda_plots = Plots(raw_data, 'Diabetes_binary')

# Paso 3: Aplicar transformaciones
temp_data = modelo.explore_different_transformations(raw_data)
transformed_data = modelo.apply_transformations(raw_data)

# Paso 4: Visualizaciones del EDA
eda_plots.plot_distributions_and_pca(raw_data, temp_data)  # Gráficos de distribuciones y PCA
eda_plots.plot_numeric_distributions()  # Distribución de variables numéricas
eda_plots.plot_binary_counts()  # Conteo de variables binarias
eda_plots.plot_boxplots()  # Diagramas de caja por variable objetivo
eda_plots.plot_crosstab()  # Tablas cruzadas de variables categóricas
eda_plots.plot_correlation_heatmap()  # Heatmap de correlación

# Paso 5: Análisis de componentes principales (PCA)
final_pca_data = modelo.apply_pca(transformed_data)

# Opcional: imprimir el DataFrame final con componentes principales
print(final_pca_data.head())
