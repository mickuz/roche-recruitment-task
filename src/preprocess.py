def impute_missing_values(df, cat_columns=[], num_columns=[]):

    for column in cat_columns:
        df[column] = df[column].fillna(df[column].value_counts().idxmax())

    for column in num_columns:
        df[column] = df[column].fillna(df[column].median())

    return df
