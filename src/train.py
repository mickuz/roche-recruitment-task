import pandas as pd
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, cross_validate

from preprocess import impute_missing_values
from build_features import convert_features, encode_features, add_new_features


df = pd.read_csv("../data/train.csv", sep=";")
df = df.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin"], axis=1)

df = impute_missing_values(df, cat_columns=["Embarked"], num_columns=["Age"])
df = convert_features(df)
df = encode_features(df, columns=["Pclass", "Age", "Embarked"])
df = add_new_features(df)

X = df.drop(["Survived"], axis=1)
y = df["Survived"]

clf = RandomForestClassifier(max_depth=5,
                             min_samples_leaf=5,
                             min_samples_split=12,
                             n_estimators=10)

clf.fit(X, y)

kfold = KFold(n_splits=10, shuffle=True)
scoring = {"accuracy": "accuracy",
           "f1": "f1_macro"}
scores = cross_validate(clf, X, y, cv=kfold, scoring=scoring)
scores = {key: score.mean() for key, score in scores.items()}

print("Accuracy: {}".format(str(scores["test_accuracy"] * 100)))
print("F1 score: {}".format(str(scores["test_f1"] * 100)))

with open("../data/model.pkl", "wb") as model_pickle:
    pkl.dump(clf, model_pickle)
