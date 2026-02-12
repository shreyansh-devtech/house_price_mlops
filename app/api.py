# ============================================
# api.py
# ============================================

# Import FastAPI framework
from fastapi import FastAPI

# Import BaseModel to define request body structure
from pydantic import BaseModel

# Import joblib to load saved model
import joblib

# Import os to check file existence
import os

# Import numpy for array conversion
import numpy as np


# -------------------------------------------------
# Step 1: Create FastAPI app instance
# -------------------------------------------------
app = FastAPI(title="House Price Classification API")


# -------------------------------------------------
# Step 2: Load trained model
# -------------------------------------------------

# Function to load best model from models folder
def load_model():
    """
    This function searches for a .pkl file inside models folder
    and loads it.
    """

    model_folder = "models"

    # List all files in models directory
    model_files = [f for f in os.listdir(model_folder) if f.endswith(".pkl")]

    if not model_files:
        raise FileNotFoundError("No trained model found in models folder.")

    # Load first model found
    model_path = os.path.join(model_folder, model_files[0])
    print(f"Loading model: {model_path}")

    model = joblib.load(model_path)

    return model


# Load model when API starts
model = load_model()


# -------------------------------------------------
# Step 3: Define Input Schema
# -------------------------------------------------

# This defines how input JSON must look
class HouseInput(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: int
    guestroom: int
    basement: int
    hotwaterheating: int
    airconditioning: int
    parking: int
    prefarea: int
    furnishingstatus: int


# -------------------------------------------------
# Step 4: Class label mapping
# -------------------------------------------------

# Convert numeric class to human-readable label
class_labels = {
    0: "Low",
    1: "Medium-Low",
    2: "Medium",
    3: "High",
    4: "Luxury"
}


# -------------------------------------------------
# Step 5: Root Endpoint
# -------------------------------------------------
@app.get("/")
def home():
    """
    Simple test endpoint to check if API is running.
    """
    return {"message": "House Price Classification API is running."}


# -------------------------------------------------
# Step 6: Prediction Endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(data: HouseInput):
    """
    This endpoint receives house details
    and returns predicted price category.
    """

    # Convert input data into list
    input_data = [
        data.area,
        data.bedrooms,
        data.bathrooms,
        data.stories,
        data.mainroad,
        data.guestroom,
        data.basement,
        data.hotwaterheating,
        data.airconditioning,
        data.parking,
        data.prefarea,
        data.furnishingstatus
    ]

    # Convert to numpy array and reshape
    input_array = np.array(input_data).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_array)[0]

    # Get probability for each class
    probabilities = model.predict_proba(input_array)[0]

    # Convert numpy floats to normal floats
    probabilities = probabilities.tolist()

    # Return response as JSON
    return {
        "predicted_class_number": int(prediction),
        "predicted_class_label": class_labels[int(prediction)],
        "class_probabilities": probabilities
    }
