from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, make_scorer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from sklearn.pipeline import make_pipeline

class Modelos:
    def __init__(self, df):
        self.df = df
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def split_data(self, use_pca=False):
        if use_pca:
            X = self.df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']]
        else:
            X = self.df.drop(['Diabetes_binary'], axis='columns')
        
        y = self.df['Diabetes_binary']
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, train_size=0.80, random_state=10)

    def logistic_regression(self, use_pca=False):
        self.split_data(use_pca)
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_val)
        self.evaluate_model(predictions)

    def linear_svc(self):
        self.split_data(use_pca=True)
        model = make_pipeline(StandardScaler(), LinearSVC(random_state=42, dual=False, max_iter=5000))
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_val)
        self.evaluate_model(predictions)

    def grid_search_svc(self):
        self.split_data(use_pca=True)
        pipeline = make_pipeline(StandardScaler(), LinearSVC(dual=False, random_state=42))
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__loss': ['hinge', 'squared_hinge'],
            'svm__tol': [1e-3, 1e-4, 1e-5],
            'svm__max_iter': [1000, 2000, 5000]
        }
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1': make_scorer(f1_score, average='weighted')
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring, refit='f1', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)

        print("Best parameters:", grid_search.best_params_)
        print("Best F1-score:", grid_search.best_score_)

    def evaluate_model(self, predictions):
        print(f"Accuracy: {accuracy_score(self.y_val, predictions):.4f}")
        print(f"Precision: {precision_score(self.y_val, predictions):.4f}")
        print(f"Recall: {recall_score(self.y_val, predictions):.4f}")
        print(f"F1-score: {f1_score(self.y_val, predictions):.4f}")

    def get_models_underoversampling(self):
        modelos, nombres = list(), list()
        modelos.append(RandomOverSampler())
        nombres.append('RandOver')
        modelos.append(TomekLinks())
        nombres.append('TomekLinks')
        modelos.append(SMOTE())
        nombres.append('SMOTE')
        modelos.append(SMOTEENN())
        nombres.append('SMOTEENN')
        return modelos, nombres

    def evaluate_under_oversampling(self):
        modelosOU, nombres = self.get_models_underoversampling()
        resultados = list()

        for i in range(len(modelosOU)):
            model = LogisticRegression()    
            kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
            pipe = make_pipeline(modelosOU[i], model)

            metrics = {
                'accuracy': make_scorer(accuracy_score),
                'recall': make_scorer(recall_score),
                'f1_score': make_scorer(f1_score),
                'precision': make_scorer(precision_score)
            }

            resultadosOU = cross_validate(pipe, self.X_train, self.y_train, scoring=metrics, cv=kfold)
            resultados.append(resultadosOU) 
            print('%s:\nmean Accuracy: %.3f (%.4f)\nmean Recall: %.3f (%.4f)\nmean F1_score: %.3f (%.4f)\nmean Precision: %.3f (%.4f)\n' % (
                nombres[i],
                np.mean(resultadosOU['test_accuracy']),
                np.std(resultadosOU['test_accuracy']), 
                np.mean(resultadosOU['test_recall']),
                np.std(resultadosOU['test_recall']),
                np.mean(resultadosOU['test_f1_score']),
                np.std(resultadosOU['test_f1_score']), 
                np.mean(resultadosOU['test_precision']),
                np.std(resultadosOU['test_precision'])
            ))

# Uso de la clase
# df = tu_dataframe_con_los_datos
modelos = Modelos(df)
modelos.logistic_regression(use_pca=False)
modelos.linear_svc()
modelos.grid_search_svc()
modelos.evaluate_under_oversampling()