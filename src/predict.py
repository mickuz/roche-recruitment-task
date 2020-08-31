import pandas as pd
import pickle as pkl

from sklearn.metrics import accuracy_score, f1_score

from preprocess import impute_missing_values
from build_features import convert_features, encode_features, add_new_features


df = pd.read_csv("../data/val.csv", sep=";")
df = df.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin"], axis=1)

df = impute_missing_values(df, cat_columns=["Embarked"], num_columns=["Age"])
df = convert_features(df)
df = encode_features(df, columns=["Pclass", "Age", "Embarked"])
df = add_new_features(df)

X = df.drop(["Survived"], axis=1)
y = df["Survived"]

with open("../data/model.pkl", "rb") as model_unpickle:
    model = pkl.load(model_unpickle)

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    print("Accuracy is: {}".format(accuracy))
    print("F1 is: {}".format(f1))
