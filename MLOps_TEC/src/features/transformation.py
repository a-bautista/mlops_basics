import numpy as np
from sklearn.preprocessing import PowerTransformer, LabelEncoder, StandardScaler

class Transformation:
    def __init__(self, data):
        self.data = data
        self.codification(data)
        self.transform(data)

    def codification(self, data):
        label_encoder = LabelEncoder()
        data['GenHlth'] = label_encoder.fit_transform(data['GenHlth'])
        data['Age'] = label_encoder.fit_transform(data['Age'])
        data['Education'] = label_encoder.fit_transform(data['Education'])
        data['Income'] = label_encoder.fit_transform(data['Income'])
        print(data[['GenHlth', 'Age', 'Education','Income']])
        return data

    def transform(self, data):
        pt = PowerTransformer(method='yeo-johnson')
        data[['BMI', 'MentHlth', 'PhysHlth']] = pt.fit_transform(data[['BMI', 'MentHlth', 'PhysHlth']])
        scaler = StandardScaler()
        data[['BMI', 'MentHlth', 'PhysHlth']] = scaler.fit_transform(data[['BMI', 'MentHlth', 'PhysHlth']])
        print(data[['BMI', 'MentHlth', 'PhysHlth']])
        return data
