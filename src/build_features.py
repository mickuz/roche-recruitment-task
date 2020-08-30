import pandas as pd


def extract_features(input_file, output_file, force_write=True):
    """Prepares the data for modeling by building new features and transforming the existing ones.

    Args:
        input_file (str): input file in the .csv format.
        output_file (str): output file in the .csv format.
        force_write (boolean): should the resulting dataframe be saved to the output file?
    """
    df = pd.read_csv(input_file, sep=";")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
    df["Embarked"] = df["Embarked"].map({"S": 1, "C": 2, "Q": 3, float("nan"): 4}).astype(int)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

    if force_write:
        df.to_csv(output_file, sep=";", index=False)
