# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd

from joblib import dump
from sklearn.model_selection import train_test_split
from .ml.data import process_data
from .ml.model import train_model, inference, compute_model_metrics  

DATA_PATH = "../data/clean_census.csv"
MODEL_PATH = "../model/random_forest_model.pkl"

# Add code to load in the data.
def load_data(data_path):
    data = pd.read_csv(data_path, index_col=None)

    train, test = train_test_split(data, test_size=0.20)
    
    return train, test
# Optional enhancement, use K-fold cross validation instead of a train-test split.

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def train_model(train, model_path, cat_features, label="salary"):

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )

    # Train and save a model.
    model = train_model(X_train, y_train)

    dump((model, encoder, lb), model_path)

def batch_inference(test_data, model_path, cat_features, label="salary"):
    pass

    # Proces the test data with the process_data function.

if __name__ == "__main__":
    train, test = load_data(DATA_PATH)
    train_model(train, MODEL_PATH, cat_features)
