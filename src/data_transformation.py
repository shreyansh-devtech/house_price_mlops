# ====================================
# data_transformation.py
# ====================================

import pandas as pd
import os


# --------------------------------------------
# Function: convert_price_to_category
# Purpose:  Convert price column into 5 classes
# --------------------------------------------
def convert_price_to_category(df):
    """
    This function converts the numerical price column
    into categorical classes based on predefined ranges.
    """

    # Define price bins
    bins = [1750000, 3000000, 5000000, 8000000, 10000000, 13300000]

    # Define class labels
    labels = [0, 1, 2, 3, 4]

    # Create new column 'price_category'
    df["price_category"] = pd.cut(
        df["price"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    print("Price converted to classification categories.")

    return df


# --------------------------------------------
# Function: encode_categorical_columns
# Purpose:  Convert yes/no & furnishing columns to numbers
# --------------------------------------------
def encode_categorical_columns(df):
    """
    This function encodes categorical features into numeric format.
    """

    # List of binary yes/no columns
    binary_columns = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea"
    ]

    # Convert yes/no to 1/0
    for col in binary_columns:
        df[col] = df[col].map({"yes": 1, "no": 0})

    # Encode furnishingstatus using mapping
    furnishing_map = {
        "unfurnished": 0,
        "semi-furnished": 1,
        "furnished": 2
    }

    df["furnishingstatus"] = df["furnishingstatus"].map(furnishing_map)

    print("Categorical columns encoded successfully.")

    return df


# --------------------------------------------
# Function: save_transformed_data
# --------------------------------------------
def save_transformed_data(df, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_csv(save_path, index=False)

    print("Transformed data saved at:", save_path)


# --------------------------------------------
# Main execution
# --------------------------------------------
if __name__ == "__main__":

    # Load processed data
    df = pd.read_csv("data/processed/Housing_clean.csv")

    # Convert price to classification
    df = convert_price_to_category(df)

    # Encode categorical columns
    df = encode_categorical_columns(df)

    # Drop original price column (optional for classification)
    df.drop(columns=["price"], inplace=True)

    # Save final dataset
    save_transformed_data(
        df,
        "data/processed/Housing_final.csv"
    )
