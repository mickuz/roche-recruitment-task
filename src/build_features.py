import pandas as pd


def convert_features(df):
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)

    bin_labels = ["Young", "Medium", "Old"]
    df["Age"] = pd.qcut(df["Age"], q=3, labels=bin_labels)

    return df


def encode_features(df, columns=[]):
    for column in columns:
        encoded_column = pd.get_dummies(df[column], drop_first=True)
        df = pd.concat([df, encoded_column], axis=1)
        df = df.drop(column, axis=1)

    return df


def add_new_features(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

    df = df.drop(["SibSp", "Parch", "FamilySize"], axis=1)

    return df
