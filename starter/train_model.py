# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from .ml.data import process_data
from .ml.model import train_model, inference, compute_model_metrics  



# Add code to load in the data.
def load_data(data_path):
    data = pd.read_csv(data_path, index_col=None)

    train, test = train_test_split(data, test_size=0.20)
    
    return train, test
# Optional enhancement, use K-fold cross validation instead of a train-test split.




def trainer(train, model_path, cat_features, label="salary"):

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )

    # Train and save a model.
    model = train_model(X_train, y_train)

    joblib.dump((model, encoder, lb), model_path)

def batch_inference(test_data, model_path, cat_features, label="salary"):
    model, encoder, lb = joblib.load(model_path)

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test_data,
        categorical_features=cat_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Evaluate model
    preds = inference(model=model, X=X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print('Precision:\t', precision)
    print('Recall:\t', recall)
    print('F-beta score:\t', fbeta)

    return precision, recall, fbeta

def online_inference(row_dict, model_path, cat_features):
    # load the model from `model_path`
    model, encoder, lb = joblib.load(model_path)

    row_transformed = list()
    X_categorical = list()
    X_continuous = list()

    for key, value in row_dict.items():
        mod_key = key.replace('_', '-')
        if mod_key in cat_features:
            X_categorical.append(value)
        else:
            X_continuous.append(value)

    y_cat = encoder.transform([X_categorical])
    y_conts = np.asarray([X_continuous])

    row_transformed = np.concatenate([y_conts, y_cat], axis=1)

    # get inference from model
    preds = inference(model=model, X=row_transformed)

    return '>50K' if preds[0] else '<=50K'

