import pandas as pd
from pandas import DataFrame


def one_hot_encode_columns(df: DataFrame, columns: list[str]) -> DataFrame:
    """
    One-hot encodes specified columns in a pandas DataFrame, storing the result as a vector in a single column.

    Args:
        df (pd.DataFrame): The original DataFrame.
        columns (list of str): List of column names to one-hot encode.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns one-hot encoded as vectors.
    """
    # Ensure the columns exist in the DataFrame
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

    # Process each column to one-hot encode
    for col in columns:
        # Apply one-hot encoding to the specific column
        encoded = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)

        # Convert encoded DataFrame to a list of lists (vectors) and add as a new column
        df[col] = encoded.apply(lambda x: list(x), axis=1)

        # Optionally, drop the original column if you don't need it anymore
        # df.drop(col, axis=1, inplace=True)

    return df


def clean_location(df: DataFrame, location_column: str) -> DataFrame:
    """
    Cleans a location column in a DataFrame by separating city and state into two different columns.

    Args:
        df (pd.DataFrame): The original DataFrame.
        location_column (str): The name of the column containing the location data.

    Returns:
        pd.DataFrame: A DataFrame with the location column split into city and state columns.
    """
    if location_column not in df.columns:
        raise ValueError(f"Column '{location_column}' not found in the DataFrame.")

    df[location_column] = df[location_column].apply(lambda x: x.split(',')[1].strip() if pd.notnull(x) and ',' in x else None)

    return df


if __name__ == "__main__":
    data = {
        'Company': ['Google', 'Facebook', 'Apple'],
        'Location': ['Mountain View, CA', 'Menlo Park, CA', 'Cupertino, CA'],
        'Role': ['Engineer', 'Designer', 'Engineer']
    }
    df = pd.DataFrame(data)

    clean_location(df, "Location")

    # Get the new DataFrame with one-hot encoded columns as vectors
    df_encoded = one_hot_encode_columns(df, ['Company', 'Role'])
    print(df_encoded)
