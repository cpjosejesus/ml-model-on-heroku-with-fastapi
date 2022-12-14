# Put the code for your API here.

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from starter.train_model import online_inference
import os

app = FastAPI()

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class RowData(BaseModel):
    age: int = Field(..., example=32)
    workclass: str = Field(..., example="Private")
    fnlwgt: int = Field(..., example=205019)
    education: str = Field(..., example="Assoc-acdm")
    education_num: int = Field(..., example=12)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Sales")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=50)
    native_country: str = Field(..., example="United-States")


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


@app.get("/")
def home():
    return {"Hello": "Welcome to project 3!"}


@app.post('/inference')
async def predict_income(inputrow: RowData):
    row_dict = jsonable_encoder(inputrow)
    model_path = 'model/random_forest_model.pkl'
    prediction = online_inference(row_dict, model_path, CAT_FEATURES)

    return {"income class": prediction}