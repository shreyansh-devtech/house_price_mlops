# ============================================
# streamlit_app.py
# ============================================

# Import streamlit library
import streamlit as st

# Import requests to call FastAPI endpoint
import requests

# Import matplotlib for probability chart
import matplotlib.pyplot as plt


# -------------------------------------------------
# Step 1: Streamlit Page Configuration
# -------------------------------------------------

# Set page title and layout
st.set_page_config(
    page_title="House Price Classification",
    layout="centered"
)

st.title("🏠 House Price Classification System")
st.write("Enter house details below to predict price category.")


# -------------------------------------------------
# Step 2: Create Input Fields
# -------------------------------------------------

# Area input (slider)
area = st.number_input("Area (sq ft)", min_value=1000, max_value=20000, value=5000)

# Bedrooms
bedrooms = st.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5, 6])

# Bathrooms
bathrooms = st.selectbox("Number of Bathrooms", [1, 2, 3, 4])

# Stories
stories = st.selectbox("Number of Stories", [1, 2, 3, 4])

# Binary features
mainroad = st.selectbox("Main Road Access", ["Yes", "No"])
guestroom = st.selectbox("Guest Room", ["Yes", "No"])
basement = st.selectbox("Basement", ["Yes", "No"])
hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
prefarea = st.selectbox("Preferred Area", ["Yes", "No"])

# Parking
parking = st.selectbox("Parking Spaces", [0, 1, 2, 3])

# Furnishing Status
furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["Unfurnished", "Semi-Furnished", "Furnished"]
)


# -------------------------------------------------
# Step 3: Convert UI Inputs to Model Format
# -------------------------------------------------

# Convert Yes/No to 1/0
binary_map = {"Yes": 1, "No": 0}

mainroad = binary_map[mainroad]
guestroom = binary_map[guestroom]
basement = binary_map[basement]
hotwaterheating = binary_map[hotwaterheating]
airconditioning = binary_map[airconditioning]
prefarea = binary_map[prefarea]

# Convert furnishing to numeric
furnishing_map = {
    "Unfurnished": 0,
    "Semi-Furnished": 1,
    "Furnished": 2
}

furnishingstatus = furnishing_map[furnishingstatus]


# -------------------------------------------------
# Step 4: Predict Button
# -------------------------------------------------

if st.button("Predict Price Category"):

    # Create JSON payload
    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }

    try:
        # Send POST request to FastAPI
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=input_data
        )

        # Convert response to JSON
        result = response.json()

        # Display predicted class
        st.success(
            f"Predicted Category: {result['predicted_class_label']}"
        )

        # Show class probabilities
        probabilities = result["class_probabilities"]

        st.subheader("Class Probability Distribution")

        # Create bar chart
        fig, ax = plt.subplots()
        ax.bar(
            ["Low", "Medium-Low", "Medium", "High", "Luxury"],
            probabilities
        )
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])

        st.pyplot(fig)

    except Exception as e:
        st.error("Error connecting to API. Make sure FastAPI is running.")
        st.write(str(e))
