class ExploreData:
    def __init__(self, data, filepath):
        self.data = data
        self.filepath = filepath
        self.explore_data(data, filepath)

    def explore_data(self, data, filepath):
        print(filepath.metadata)
        print(filepath.variables)
        print(data.head())
        print(data.describe().T)
        print(data.info())
        print(data.shape)
        print(data.isnull().sum())
        
        binary_columns = [col for col in data.columns if set(data[col].unique()).issubset({0, 1})]
        for col in binary_columns:
            data[col] = data[col].astype('bool')

        categorical_columns = ['GenHlth', 'Age', 'Education', 'Income']
        for col in categorical_columns:
            data[col] = data[col].astype('category')
        
        print(data.dtypes)
