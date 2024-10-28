from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class process_model:
    def __init__(self):
        self.instance_uo, self.modelos, self.nombres, self.params = self.get_models_underoversampling()
        
    def get_models_underoversampling(self):
        # Define las instancias de sobremuestreo/submuestreo y modelos a usar
        instance_uo, modelos, nombres = list(), list(), list()
        dict_uo = {1: RandomOverSampler(), 2: TomekLinks(), 3: SMOTE(), 4: SMOTEENN()}
        dict_nombres = {0: 'Logistic Regression', 1: 'Log_RandOver', 2: 'Log_TomekLinks', 3: 'Log_SMOTE', 4: 'Log_SMOTEENN', 5: 'RandomForest'}
        dict_params = {0: {'n_estimators': 200, 'random_state': 42}, 1: {'class_weight': 'balanced'}}
        
        for i in range(6):
            # Asigna sobremuestreo/submuestreo a cada modelo
            instance_uo.append(dict_uo.get(i, np.NaN))
            # Define el modelo correspondiente
            if i == 0:
                modelos.append(LogisticRegression())
                params = np.NaN
            elif i == 5:
                params = dict_params.get(0)
                modelos.append(RandomForestClassifier(**params))
            else:
                params = dict_params.get(1)
                modelos.append(LogisticRegression(**params))
            nombres.append(dict_nombres.get(i))
            
        return instance_uo, modelos, nombres, params

    def create_pipeline(self, model, inst_uo):
        # Crea un pipeline con el modelo y la técnica de sobremuestreo/submuestreo si es necesario
        if isinstance(inst_uo, float):
            return make_pipeline(model)
        else:
            return make_pipeline(inst_uo, model)

    def cross_validate_model(self, model_pipeline, X_train, y_train):
        # Realiza validación cruzada
        metrics = ['accuracy', 'recall', 'f1']
        kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
        return cross_validate(model_pipeline, X_train, y_train, scoring=metrics, cv=kfold)

    def train_and_evaluate(self, X_train, X_val, y_train, y_val):
        for inst_uo, model, name in zip(self.instance_uo, self.modelos, self.nombres):
            model_pipeline = self.create_pipeline(model, inst_uo)
            resultados = self.cross_validate_model(model_pipeline, X_train, y_train)

            # Entrena el modelo con los mejores resultados
            model_pipeline.fit(X_train, y_train)
            predictions = model_pipeline.predict(X_val)
            
            # Registro del modelo en MLflow
            self.register_model_in_mlflow(model_pipeline, name, resultados)
            
            # Genera y guarda la matriz de confusión
            process_model.model_statistics.mi_cm(y_val, predictions, name)

    def register_model_in_mlflow(self, model_pipeline, name, resultados):
        # Registra métricas y parámetros en MLflow
        mean_accuracy = np.mean(resultados['test_accuracy'])
        std_accuracy = np.std(resultados['test_accuracy'])
        mean_recall = np.mean(resultados['test_recall'])
        std_recall = np.std(resultados['test_recall'])
        mean_f1 = np.mean(resultados['test_f1'])
        std_f1 = np.std(resultados['test_f1'])
        
        with mlflow.start_run(run_name=name):
            mlflow.log_params(self.params)
            mlflow.log_metrics({
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "mean_recall": mean_recall,
                "std_recall": std_recall,
                "mean_f1": mean_f1,
                "std_f1": std_f1
            })
            mlflow.sklearn.log_model(model_pipeline, artifact_path="models")

    class model_statistics:
        # Clase para métricas y matriz de confusión
        @staticmethod
        def mi_cm(yreal, ypred, name):
            cm = confusion_matrix(yreal, ypred)
            text = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
            vf = ['( TN )', '( FP )', '( FN )', '( TP )']
            freq = ["{0:0.0f}".format(value) for value in cm.flatten()]
            percent = ["{0:.1%}".format(value) for value in cm.flatten() / np.sum(cm)]
            
            labels = [f"{v1}\n{v2}\n{v3}\n{v4}" for v1, v2, v3, v4 in zip(text, vf, freq, percent)]
            labels = np.asarray(labels).reshape(2, 2)
            
            plt.figure(figsize=(6, 4))
            ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Spectral', cbar=False)
            ax.set(ylabel="Real labels", xlabel="Prediction labels")
            name_confusion_matrix = f"confusion_matrix_{name}.png"
            plt.savefig(name_confusion_matrix)
            plt.show()

# Uso de la clase
process = process_model()
process.train_and_evaluate(X_train, X_val, y_train, y_val)
