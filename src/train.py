import pandas as pd
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess import clean_data
from build_features import extract_features


extract_features("../data/train.csv", "../data/train_bf.csv")
clean_data("../data/train_bf.csv", "../data/train_bf.csv")

df = pd.read_csv("../data/train_bf.csv", sep=";")

y = df["Survived"]
X = df.drop("Survived", axis=1)

clf = RandomForestClassifier(n_estimators=10)

clf.fit(X, y)
predictions = clf.predict(X)
metric_name = "train_accuracy"
metric_result = accuracy_score(y, predictions)

with open("../data/model.pkl", "wb") as model_pickle:
    pkl.dump(clf, model_pickle)

info = "{} for the model is {}".format(metric_name, str(metric_result))
print(info)
