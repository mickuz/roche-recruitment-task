import pandas as pd


def clean_data(input_file, output_file):
    """Removes unnecessary features and null values from the data.

    Args:
        input_file (str): input file in the .csv format.
        output_file (str): output file in the .csv format.
    """
    data = pd.read_csv(input_file, sep=";")
    data = data.dropna()
    data = data.drop(["Name", "Ticket", "Cabin"], axis=1)

    data.to_csv(output_file, sep=";", index=False)

