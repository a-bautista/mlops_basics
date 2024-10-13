from sklearn.model_selection import train_test_split
import os
import yaml  # Make sure you have this import at the top

class DataSplitter:
    def __init__(self, data):
        self.data = data

    def split_data(self):
        # Define the path to params.yaml
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'params.yaml')

        # Load parameters
        with open(file_path) as file:
            params = yaml.safe_load(file)

        # Ensure to access the correct structure in params
        X = self.data.drop(columns=['Diabetes_binary'])
        y = self.data['Diabetes_binary']

        # Split the data using the parameters
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['model']['test_size'], random_state=params['model']['random_state'])

        return X_train, X_test, y_train, y_test
