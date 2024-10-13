from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yaml

class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = RandomForestClassifier()

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy * 100:.2f}%')
