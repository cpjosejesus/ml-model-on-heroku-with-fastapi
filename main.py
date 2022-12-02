# Put the code for your API here.

from starter.train_model import load_data, trainer, batch_inference

DATA_PATH = "./data/clean_census.csv"
MODEL_PATH = "./model/random_forest_model.pkl"

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

