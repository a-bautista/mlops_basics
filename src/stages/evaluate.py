import argparse
import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from typing import Text, Dict
import yaml


def write_confusion_matrix_data(y_true, predicted, filename):
    assert len(predicted) == len(y_true)
    cf = pd.DataFrame(list(zip(y_true, predicted)), columns=["y_true", "predicted"])
    cf.to_csv(filename, index=False)

def evaluate_model(config_path: Text) -> None:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    print('Log: Load model')
    model_path = config['train']['model_path']
    model = joblib.load(model_path)

    print('Log: Load X_test and y_test')
    X_test = pd.read_csv(config['data_split']['testset_path'])
    y_test = pd.read_csv(config['data_split']['test_target'])

    print('Log: Evaluate (build report)')
    prediction = model.predict(X_test)
    f1 = f1_score(y_true=y_test, y_pred=prediction, average='macro')

    labels = load_iris(as_frame=True).target_names.tolist()

    cm = confusion_matrix(prediction, y_test)
    report = {
        'f1': f1,
        'cm': cm,
        'actual': y_test,
        'predicted': prediction
    }

    print('Log: Save metrics')
    # Guardar archivo con métrica f1
    reports_folder = Path(config['evaluate']['reports_dir'])
    metrics_path = reports_folder / config['evaluate']['metrics_file']

    json.dump(
        obj={'f1_score': report['f1']},
        fp=open(metrics_path, 'w')
    )

    # Asegúrate de que y_test sea un arreglo de 0 y 1
    # Si y_test es un DataFrame, extraer la columna como un arreglo
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.values.flatten()  # Aplana a un arreglo unidimensional

    # Guardar archivo con datos CSV para los plots de dvc
    confusion_matrix_data_path = reports_folder / config['evaluate']['confusion_matrix_data']
    write_confusion_matrix_data(y_test, prediction, filename=confusion_matrix_data_path)

    print(f'F1 metrics file saved to : {metrics_path}')

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)