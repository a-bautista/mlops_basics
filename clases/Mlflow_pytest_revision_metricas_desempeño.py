#en esta clase buscamos encontrar algun modelo cuyo recall con promedio ponderado (que es la metrica que hemos definido como la principal para elegir nuestros modelos) 
# sea menor al benchmark establecido (70%) y asi analizar su posible depuracion al no tener el minimo desempeño esperado
%%ipytest
import ipytest
import pytest
import mlflow
from mlflow import MlflowClient

@pytest.fixture()
def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000") 
    experiment_id = client.get_experiment_by_name("Diabetes_Diagnostic").experiment_id
    return experiment_id
def test_model_registration(setup_mlflow):
    experiment_id = setup_mlflow 
    for run in client.search_runs(experiment_ids=experiment_id):
        recall_avg_weight=run.data.metrics['recall_avg_weight']
        modelo=run.data.tags['Model for diabetes diagnostic']
        assert recall_avg_weight >  0.70, f"Modelo:{modelo}, Recall Promedio ponderado igual a {recall_avg_weight:.2f}, se espera mayor a 0.70."