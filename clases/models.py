import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler

class process_model:
    def __init__(self):
        pass

    class model_statistics:
        def __init__(self, yreal, ypred):
            self.yreal = yreal
            self.ypred = ypred
            self.my_accuracy(yreal, ypred)
            self.my_recall(yreal, ypred)
            self.my_gmean(yreal, ypred)
            self.my_precision(yreal, ypred)
            self.my_f1_score(yreal, ypred)
            self.mi_cm(yreal, ypred, "default")

        def my_accuracy(self, yreal, ypred):
            vn, fp, fn, vp = confusion_matrix(yreal, ypred).ravel()
            tot = confusion_matrix(yreal, ypred).sum()
            return (vp + vn) / tot

        def my_recall(self, yreal, ypred):
            vn, fp, fn, vp = confusion_matrix(yreal, ypred).ravel()
            return vp / (vp + fn)

        def my_gmean(self, yreal, ypred):
            vn, fp, fn, vp = confusion_matrix(yreal, ypred).ravel()
            especificidad = vn / (vn + fp)
            recall = self.my_recall(yreal, ypred)
            return np.sqrt(recall * especificidad)

        def my_precision(self, yreal, ypred):
            vn, fp, fn, vp = confusion_matrix(yreal, ypred).ravel()
            return vp / (vp + fp)

        def my_f1_score(self, yreal, ypred):
            vn, fp, fn, vp = confusion_matrix(yreal, ypred).ravel()
            return (2 * vp) / (2 * vp + fp + fn)

        def mi_cm(self, yreal, ypred, name):
            cm = confusion_matrix(yreal, ypred)
            text = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
            vf = ['( TN )', '( FP )', '( FN )', '( TP )']
            freq = ["{0:0.0f}".format(value) for value in cm.flatten()]
            percent = ["{0:.1%}".format(value) for value in cm.flatten() / np.sum(cm)]
            labels = [f"{v1}\n{v2}\n{v3}\n{v4}" for v1, v2, v3, v4 in zip(text, vf, freq, percent)]
            labels = np.asarray(labels).reshape(2, 2)
            
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=labels, fmt='', cmap='Spectral', cbar=False)
            plt.ylabel("Real labels")
            plt.xlabel("Prediction labels")
            plt.savefig(f"confusion_matrix_{name}.png")
            plt.show()

    class list_models:
        def __init__(self, X_train, y_train, X_val, y_val):
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.instance_uo, self.modelos, self.nombres, self.params_list = self.get_models_underoversampling()
            self.creation_models()

        def get_models_underoversampling(self):
            instance_uo, modelos, nombres = list(), list(), list()
            dict_uo = {1: RandomOverSampler(), 2: TomekLinks(), 3: SMOTE(), 4: SMOTEENN()}
            dict_nombres = {0: 'Log', 1: 'Log_RandOver', 2: 'Log_TomekLinks', 3: 'Log_SMOTE', 4: 'Log_SMOTEENN', 5: 'RandForest'}
            dict_params = {0: {'n_estimators': 200, 'random_state': 42}, 1: {'class_weight': 'balanced'}}
            
            for i in range(6):
                if i in [0, 5]:
                    instance_uo.append(np.nan)
                else:
                    instance_uo.append(dict_uo.get(i))
                if i == 0:
                    modelos.append(LogisticRegression())
                    params = np.nan
                elif i == 5:
                    params = dict_params.get(0)
                    modelos.append(RandomForestClassifier(**params))
                else:
                    params = dict_params.get(1)
                    modelos.append(LogisticRegression(**params))
                nombres.append(dict_nombres.get(i))
            return instance_uo, modelos, nombres, params

        def get_model_params(self, model):
            # Define aquí cómo obtener los parámetros del modelo.
            params = {}  # Inicializa como un diccionario vacío
            if isinstance(model, LogisticRegression):
                params = {'class_weight': 'balanced'}
            elif isinstance(model, RandomForestClassifier):
                params = {'n_estimators': 200, 'random_state': 42}
            # Puedes agregar más condiciones para otros modelos
            return params  # Asegúrate de que esto siempre devuelva un diccionario

        #def registation_models(self, name, model):
            #with mlflow.start_run(run_name=name):
                #mlflow.log_params(self.params)
                #mlflow.log_metrics({
                    #"mean_accuracy": self.mean_accuracy,
                    #"std_accuracy": self.std_accuracy,
                    #"mean_recall": self.mean_recall,
                    #"std_recall": self.std_recall,
                    #"mean_f1": self.mean_f1,
                    #"std_f1": self.std_f1
                #})
                #mlflow.sklearn.log_model(model, artifact_path="models")

        def creation_models(self):
            for inst_uo, model, name in zip(self.instance_uo, self.modelos, self.nombres):
                resultados = []

                # Si inst_uo es None o np.nan, usa 'passthrough' en el pipeline
                if inst_uo is None or isinstance(inst_uo, float) and np.isnan(inst_uo):
                    model_pipeline = make_pipeline(model)
                else:
                    # Usa el transformador inst_uo en el pipeline
                    model_pipeline = make_pipeline(inst_uo, model)

                metrics = ['accuracy', 'recall', 'f1']
                kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
                resultadosOU = cross_validate(model_pipeline, self.X_train, self.y_train, scoring=metrics, cv=kfold)
                resultados.append(resultadosOU)

                # Calcular estadísticas
                self.mean_accuracy = np.mean(resultadosOU['test_accuracy'])
                self.std_accuracy = np.std(resultadosOU['test_accuracy'])
                self.mean_recall = np.mean(resultadosOU['test_recall'])
                self.std_recall = np.std(resultadosOU['test_recall'])
                self.mean_f1 = np.mean(resultadosOU['test_f1'])
                self.std_f1 = np.std(resultadosOU['test_f1'])

                # Entrenar y evaluar en conjunto de validación
                model_pipeline.fit(self.X_train, self.y_train)
                predictions = model_pipeline.predict(self.X_val)

                # Asignar los parámetros para el modelo actual
                self.params = self.get_model_params(model)

                # Registrar los modelos
                #self.registation_models(name, model)

                ## se manda llamar a la clase de mlflow.py

                # Estadísticas del modelo
                stats = process_model.model_statistics(self.y_val, predictions)
                stats.mi_cm(self.y_val, predictions, name)
