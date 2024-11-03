import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment("Diabetes_Diagnostic_test")


class registration_models:
    def __init__(self):
        self.mean_accuracy = None
        self.std_accuracy = None
        self.mean_recall = None
        self.std_recall = None
        self.mean_f1 = None
        self.std_f1 = None
        self.recall_avg_weight = None
        self.f1_avg_weight = None

    def registration_models(self, name, model, param):
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("Model for diabetes diagnostic", name)
            print(param)
            if not isinstance(param, float):
                mlflow.log_params(param)

            # Comprobar si las métricas son None y establecer un valor predeterminado si lo son
            self.mean_accuracy = self.mean_accuracy if self.mean_accuracy is not None else 0.0
            self.std_accuracy = self.std_accuracy if self.std_accuracy is not None else 0.0
            self.mean_recall = self.mean_recall if self.mean_recall is not None else 0.0
            self.std_recall = self.std_recall if self.std_recall is not None else 0.0
            self.mean_f1 = self.mean_f1 if self.mean_f1 is not None else 0.0
            self.std_f1 = self.std_f1 if self.std_f1 is not None else 0.0
            # 02112024 se agrega para tener las metricas de recall y f1 score ponderados
            self.recall_avg_weight = self.recall_avg_weight if self.recall_avg_weight is not None else 0.0
            self.f1_avg_weight = self.f1_avg_weight if self.f1_avg_weight is not None else 0.0

            mlflow.log_metrics({
                "mean_accuracy": self.mean_accuracy,
                "std_accuracy": self.std_accuracy,
                "mean_recall": self.mean_recall, "std_recall": self.std_recall,
                "mean_f1": self.mean_f1, "std_f1": self.std_f1,
                # 02112024 se agrega para tener las metricas de recall y f1 score ponderados
                "recall_avg_weight": self.recall_avg_weight,
                "f1_avg_weight": self.f1_avg_weight
            })
            # Log the model
            mlflow.sklearn.log_model(model, artifact_path="models")
