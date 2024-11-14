from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ucimlrepo import fetch_ucirepo
from eda import Diabetes
from eda_plots import Plots
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from models import process_model
from MLFlow import registration_models

# Ignorar advertencias
warnings.filterwarnings("ignore")

# Inicializar FastAPI
app = FastAPI()

# No es necesario el modelo TransformData si no vamos a pedir id en la solicitud
# class TransformData(BaseModel):
#     id: int

class ModelResponse(BaseModel):
    mean_accuracy: float
    std_accuracy: float
    mean_recall: float
    std_recall: float
    mean_f1: float
    std_f1: float
    recall_avg_weight: float
    f1_avg_weight: float

# Función de carga y preparación de datos con id fijo
def cargar_y_preparar_datos():
    id_fijo = 891  # Aquí defines el id fijo
    cdc_diabetes_health_indicators = fetch_ucirepo(id=id_fijo)
    modelo = Diabetes(cdc_diabetes_health_indicators)
    raw_data = modelo.load_data()

    # Usar solo el 10% del dataset para pruebas, por ejemplo
    raw_data = raw_data.sample(frac=0.1, random_state=42)

    modelo.explore_data(raw_data)
    modelo.preprocess_conversion_cols(raw_data)
    plots = Plots(raw_data, "Diabetes_binary")
    #plots.summary_statistics()
    #plots.plot_numeric_distributions()
    #plots.plot_binary_counts()
    #plots.plot_boxplots()
    #plots.plot_crosstab()
    #plots.plot_correlation_heatmap()
    return modelo, raw_data

# Endpoint para cargar y transformar datos (sin solicitar id)
@app.post("/load_and_transform")
def load_and_transform():
    try:
        modelo, raw_data = cargar_y_preparar_datos()
        
        # Limitar transformaciones
        temp_data = modelo.explore_different_transformations(raw_data)
        transformed_data = modelo.apply_transformations(raw_data)

        if 'Diabetes_binary' not in transformed_data.columns:
            return {"detail": "Advertencia: 'Diabetes_binary' no está presente después de las transformaciones."}

        diabetes_binary_column = transformed_data['Diabetes_binary']
        df_pca, explained_variance = modelo.apply_pca(transformed_data)
        df_pca['Diabetes_binary'] = diabetes_binary_column

        df = modelo.true_false_to_one_hot(df_pca)
        X_train, X_val, y_train, y_val = modelo.split_data(df)

        # Convertir a formato serializable y limpiar memoria
        X_train_dict = X_train.to_dict(orient="records")
        X_val_dict = X_val.to_dict(orient="records")
        
        # Liberar variables temporales para evitar alto uso de memoria
        del temp_data, transformed_data, df_pca, df, X_train, X_val

        return {"detail": "Transformación completa", "X_train": X_train_dict, "X_val": X_val_dict}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error desconocido: {str(e)}")




# Endpoint para entrenar y evaluar el modelo con id fijo
@app.post("/train_model", response_model=ModelResponse)
def train_model():
    try:
        # Usar el id fijo aquí (por ejemplo el id 891)
        id_fijo = 891
        modelo, raw_data = cargar_y_preparar_datos()
        transformed_data = modelo.apply_transformations(raw_data)
        diabetes_binary_column = transformed_data['Diabetes_binary']
        df_pca, explained_variance = modelo.apply_pca(transformed_data)
        df_pca['Diabetes_binary'] = diabetes_binary_column
        df = modelo.true_false_to_one_hot(df_pca)
        X_train, X_val, y_train, y_val = modelo.split_data(df)

        # Instancia y evalúa modelos
        process = process_model()
        model_evaluator = process.list_models(X_train, y_train, X_val, y_val)

        # Registro de métricas con MLFlow
        model_registration = registration_models()
        model_registration.register(model_evaluator)

        # Preparar respuesta
        response = {
            "mean_accuracy": model_evaluator.registration.mean_accuracy,
            "std_accuracy": model_evaluator.registration.std_accuracy,
            "mean_recall": model_evaluator.registration.mean_recall,
            "std_recall": model_evaluator.registration.std_recall,
            "mean_f1": model_evaluator.registration.mean_f1,
            "std_f1": model_evaluator.registration.std_f1,
            "recall_avg_weight": model_evaluator.recall_avg_weight,
            "f1_avg_weight": model_evaluator.f1_avg_weight
        }
        print(response)
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Iniciar el servidor FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000,  workers=1, limit_concurrency=10)
