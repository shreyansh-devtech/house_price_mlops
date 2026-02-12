# ============================================
# test_model_exists.py
# ============================================

# Import os module to check file existence
import os


def test_model_file_exists():
    """
    This test checks whether at least one .pkl model file
    exists inside the models folder after training.
    """

    model_folder = "models"

    # Check if models directory exists
    assert os.path.exists(model_folder), "Models folder does not exist."

    # Get list of model files
    model_files = [
        f for f in os.listdir(model_folder)
        if f.endswith(".pkl")
    ]

    # Assert at least one model exists
    assert len(model_files) > 0, "No trained model file found."
