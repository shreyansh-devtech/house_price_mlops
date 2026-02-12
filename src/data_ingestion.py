# ================================
# data_ingestion.py
# ================================

# Import pandas library to handle CSV and dataframe
import pandas as pd

# Import os module to handle file paths
import os


# --------------------------------------------
# Function: load_raw_data
# Purpose:  Read raw CSV file from disk
# --------------------------------------------
def load_raw_data(file_path):
    """
    This function reads the raw dataset from the given file path.

    Parameter:
        file_path (str): Path to the CSV file

    Returns:
        df (DataFrame): Loaded pandas DataFrame
    """

    # Read CSV file
    df = pd.read_csv(file_path)

    # Print basic information
    print("Dataset Loaded Successfully!")
    print("Shape of dataset:", df.shape)
    print("\nColumns:", df.columns.tolist())

    return df


# --------------------------------------------
# Function: save_processed_data
# Purpose:  Save dataframe into processed folder
# --------------------------------------------
def save_processed_data(df, save_path):
    """
    This function saves dataframe into a new CSV file.

    Parameter:
        df (DataFrame): Data to save
        save_path (str): Path where processed file will be saved
    """

    # Create directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save dataframe as CSV
    df.to_csv(save_path, index=False)

    print(f"Processed data saved at: {save_path}")


# --------------------------------------------
# Main execution block
# --------------------------------------------
if __name__ == "__main__":

    # Define raw data path
    raw_data_path = "data/raw/Housing.csv"

    # Define processed save path
    processed_data_path = "data/processed/Housing_clean.csv"

    # Load dataset
    df = load_raw_data(raw_data_path)

    # Save dataset into processed folder
    save_processed_data(df, processed_data_path)
