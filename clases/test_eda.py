import pytest
from eda import Diabetes
from ucimlrepo import fetch_ucirepo
import pandas as pd
from main import cdc_diabetes_health_indicators
import numpy as np
from main import modelo
import matplotlib
matplotlib.use('Agg') 

@pytest.fixture
def diabetes_instance():
    return modelo

def test_load_data(diabetes_instance):
    data = diabetes_instance.load_data()
    assert data is not None, "Los datos no se cargaron correctamente"
    assert 'Diabetes_binary' in data.columns, "La columna 'Diabetes_binary' no se encontró en los datos"
    assert data.shape[1] >= 22, "El DataFrame cargado tiene menos de 8 columnas"


def test_preprocess_conversion_cols(diabetes_instance):
    data = diabetes_instance.load_data()
    diabetes_instance.preprocess_conversion_cols(data)
    assert data['GenHlth'].dtype.name.lower() == 'category', "La columna GenHlth debería ser categórica"
    assert data['Age'].dtype.name.lower() == 'category', "La columna Age debería ser categórica"



def test_explore_different_transformations(diabetes_instance):
    data = diabetes_instance.load_data()
    transformed_data = diabetes_instance.explore_different_transformations(data)
    assert not transformed_data[['Log_BMI', 'Sqrt_BMI']].isnull().any().any(), "Las columnas transformadas no deberían tener valores nulos"


def test_apply_transformations(diabetes_instance):
    data = diabetes_instance.load_data()
    transformed_data = diabetes_instance.apply_transformations(data)
    assert 'BMI' in transformed_data.columns, "La columna BMI debería estar en el DataFrame"

def test_apply_pca(diabetes_instance):
    data = diabetes_instance.load_data()
    pca_df, explained_variance = diabetes_instance.apply_pca(data)
    assert pca_df.shape[1] <= data.shape[1], "El número de componentes principales no puede exceder el número de características originales"

def test_true_false_to_one_hot(diabetes_instance):
    data = diabetes_instance.load_data()
    converted_data = diabetes_instance.true_false_to_one_hot(data)
    assert set(converted_data['Diabetes_binary'].unique()).issubset({0, 1}), "La columna Diabetes_binary debería contener solo valores 0 o 1"

def test_split_data(diabetes_instance):
    data = diabetes_instance.load_data()
    X_train, X_val, y_train, y_val = diabetes_instance.split_data(data)
    assert X_train.shape[0] > 0, "El conjunto de entrenamiento no debería estar vacío"
    assert X_val.shape[0] > 0, "El conjunto de validación no debería estar vacío"
    assert y_train.shape[0] > 0, "El conjunto de etiquetas de entrenamiento no debería estar vacío"
    assert y_val.shape[0] > 0, "El conjunto de etiquetas de validación no debería estar vacío"


if __name__ == '__main__':
    pytest.main()