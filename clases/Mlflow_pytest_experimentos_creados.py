
##en esta prueba de pytest revisamos que todos nuestras ejecuciones de modelos hayan sido creadas exitosamente en nuestro experimento
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
    assert len(runs) > 0