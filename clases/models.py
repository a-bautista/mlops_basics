import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
#from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from MLFlow import registration_models 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score,make_scorer
from sklearn.svm import LinearSVC


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


    class model_statistics_avg_weight:
        def __init__(self):
            pass
        def my_rec_weight(self,yreal,ypred):
            recall_avg_weight=recall_score(yreal, ypred, average='weighted')
            return recall_avg_weight
        def my_f1_weight(self,yreal,ypred):
            f1_avg_weight=f1_score(yreal, ypred, average='weighted')
            return f1_avg_weight


    class list_models:
        def __init__(self, X_train, y_train, X_val, y_val):
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.instance_uo, self.modelos, self.nombres, self.params_list = self.get_models_underoversampling()
            #self.registration = registration_models()
            self.registration = registration_models()
            self.creation_models()

        def get_models_underoversampling(self):
            instance_uo, modelos, nombres = list(), list(), list()
            dict_uo = {1: RandomOverSampler(), 2: TomekLinks(), 3: SMOTE(), 4: SMOTEENN()}
            #dict_nombres = {0: 'Log', 1: 'Log_RandOver', 2: 'Log_TomekLinks', 3: 'Log_SMOTE', 4: 'Log_SMOTEENN', 5: 'RandForest'}
            #02112024 se agrega modelo de SVC
            dict_nombres = {0: 'Log', 1: 'Log_RandOver', 2: 'Log_TomekLinks', 3: 'Log_SMOTE', 4: 'Log_SMOTEENN', 5: 'RandForest',6:'Linear_SVM'}
            #02112024 se agregan los hiperparametros de SVC
            dict_params={0:{'n_estimators':200,'random_state':42},1:{'random_state':42, 'dual':False, 'max_iter':5000},2:{'class_weight':'balanced'}}
            #dict_params = {0: {'n_estimators': 200, 'random_state': 42}, 1: {'class_weight': 'balanced'}}
            
            #02112024 se agrega modelo de SVC
            for i in range(7):
            #for i in range(6):
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
                elif i == 6:
                #02112024 se agrega modelo de SVC
                    params = dict_params.get(1)
                    modelos.append(LinearSVC(**params))
                else:
                    params = dict_params.get(2)
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
            elif isinstance(model, LinearSVC):
                params= {'random_state':42, 'dual':False, 'max_iter':5000}
            return params  



        def creation_models(self):
            #02112024 se agrega funcion para mandar a llamar las metricas ponderadas
            stats_avg_weight=process_model.model_statistics_avg_weight()
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
                self.registration.mean_accuracy = np.mean(resultadosOU['test_accuracy'])
                self.registration.std_accuracy = np.std(resultadosOU['test_accuracy'])
                self.registration.mean_recall = np.mean(resultadosOU['test_recall'])
                self.registration.std_recall = np.std(resultadosOU['test_recall'])
                self.registration.mean_f1 = np.mean(resultadosOU['test_f1'])
                self.registration.std_f1 = np.std(resultadosOU['test_f1'])
                
                
                # Entrenar y evaluar en conjunto de validación
                model_pipeline.fit(self.X_train, self.y_train)
                predictions = model_pipeline.predict(self.X_val)

                # Asignar los parámetros para el modelo actual
                param = self.get_model_params(model)
                #02112024 se agrega funcion para mandar a llamar las metricas ponderadas
                
                self.recall_avg_weight = stats_avg_weight.my_rec_weight(self.y_val, predictions)
                self.f1_avg_weight = stats_avg_weight.my_f1_weight(self.y_val, predictions)

                self.registration.recall_avg_weight=self.recall_avg_weight
                self.registration.f1_avg_weight=self.f1_avg_weight

                # Registrar los modelos
                #self.registation_models(name, model)
                self.registration.registration_models(name,model,param)
                ## se manda llamar a la clase de mlflow.py

                # Estadísticas del modelo
                stats = process_model.model_statistics(self.y_val, predictions)
                stats.mi_cm(self.y_val, predictions, name)