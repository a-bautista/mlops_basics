%%ipytest
import mlflow
import ipytest
import pytest
from mlflow import MlflowClient
@pytest.fixture()
def setup_mlflow():
    client = MlflowClient(tracking_uri="http://localhost:5000")
    experiment_id = client.get_experiment_by_name("Diabetes_Diagnostic").experiment_id
    return experiment_id
def test_model_registration(setup_mlflow):
    experiment_id = setup_mlflow 
    runs=client.search_runs(experiment_ids=experiment_id)
    assert len(runs) > 0 #revisamos si se registro al menos un experimento en la carga de todos los modelos

def test_metrics_versus_benchmark(setup_mlflow):
    experiment_id = setup_mlflow 
    for run in client.search_runs(experiment_ids=experiment_id):
        recall_avg_weight=run.data.metrics['recall_avg_weight']
        modelo=run.data.tags['Model for diabetes diagnostic']
        assert recall_avg_weight >  0.70, f"Modelo:{modelo}, Recall Promedio ponderado igual a {recall_avg_weight:.2f}, se espera mayor a 0.70." #revisamos si alguno de nuestros modelos tiene un recall con promedio ponderado menor al benchmark establecido (70%)