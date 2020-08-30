import pandas as pd
import pickle as pkl

from preprocess import clean_data
from build_features import extract_features


extract_features("../data/val.csv", "../data/val_bf.csv")
clean_data("../data/val_bf.csv", "../data/val_bf.csv")

df = pd.read_csv("../data/val_bf.csv", sep=";")

df.dropna(inplace=True)

target = df["Survived"].values
df = df.drop(["Survived"], axis=1)

with open("../data/model.pkl", "rb") as model_unpickle:
    model = pkl.load(model_unpickle)

predictions = model.predict(df)

accuracy = (predictions == target).mean()
print("accuracy is", accuracy)
