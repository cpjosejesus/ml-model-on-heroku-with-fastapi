"""
This train and save a model in the directory:
 - ./model/


"""
from starter.train_model import load_data, trainer, batch_inference

DATA_PATH = "./data/clean_census.csv"
MODEL_PATH = "./model/random_forest_model.pkl"
ROOT_PATH = "./model/"

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

if __name__ == '__main__':
    # Get the splitted data
    train_data, test_data = load_data(DATA_PATH)

    # Training the model on the train data
    trainer(train_data, MODEL_PATH, cat_features)

    # evaluating the model on the test data
    precision, recall, f_beta = batch_inference(test_data,
                                                MODEL_PATH,
                                                cat_features)