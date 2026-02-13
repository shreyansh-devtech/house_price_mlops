import matplotlib.pyplot as plt
import requests
import streamlit as st

API_URL_DEFAULT = "http://127.0.0.1:8000/predict"
CLASS_NAMES = ["Low", "Medium-Low", "Medium", "High", "Luxury"]

st.set_page_config(
    page_title="House Price Classification",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg-1: #f7f6f2;
        --bg-2: #f0ece2;
        --ink: #222831;
        --muted: #6b7280;
        --card: rgba(255, 255, 255, 0.85);
        --accent: #1f7a8c;
        --accent-2: #bfdbf7;
    }
    .stApp {
        background: radial-gradient(circle at top right, var(--accent-2), var(--bg-1) 45%, var(--bg-2));
    }
    .hero {
        background: linear-gradient(130deg, #1f7a8c, #2a9d8f);
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        margin-bottom: 0.8rem;
    }
    .soft-card {
        background: var(--card);
        padding: 0.7rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(31, 122, 140, 0.15);
        margin-bottom: 0.6rem;
    }
    .soft-card h4 {
        color: var(--ink);
        margin: 0;
    }
    .soft-card p {
        color: var(--muted);
        margin: 0.2rem 0 0 0;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 0;
        background: linear-gradient(90deg, #1f7a8c, #2a9d8f);
        color: white;
        font-weight: 600;
        padding: 0.55rem 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h2 style="margin:0;">House Price Classification Dashboard</h2>
        <p style="margin:0.25rem 0 0 0; opacity:0.95;">
            Advanced prediction UI with live confidence insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("Prediction API URL", value=API_URL_DEFAULT)
    st.caption("Make sure FastAPI is running before prediction.")
    st.divider()
    st.subheader("Model Inputs Guide")
    st.write("- Higher area and amenities often move prediction upward.")
    st.write("- Furnishing scale: Unfurnished (0), Semi (1), Furnished (2).")


with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="soft-card"><h4>Property Size</h4><p>Base physical attributes</p></div>', unsafe_allow_html=True)
        area = st.slider("Area (sq ft)", min_value=1000, max_value=20000, value=5000, step=100)
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
        stories = st.selectbox("Stories", [1, 2, 3, 4], index=1)
        parking = st.selectbox("Parking Spaces", [0, 1, 2, 3], index=1)

    with right:
        st.markdown('<div class="soft-card"><h4>Amenities</h4><p>Location and comfort features</p></div>', unsafe_allow_html=True)
        mainroad = st.selectbox("Main Road Access", ["Yes", "No"], index=0)
        guestroom = st.selectbox("Guest Room", ["Yes", "No"], index=1)
        basement = st.selectbox("Basement", ["Yes", "No"], index=1)
        hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"], index=1)
        airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"], index=0)
        prefarea = st.selectbox("Preferred Area", ["Yes", "No"], index=0)
        furnishingstatus = st.selectbox(
            "Furnishing Status",
            ["Unfurnished", "Semi-Furnished", "Furnished"],
            index=1,
        )

    submit = st.form_submit_button("Predict Price Category")

if submit:
    binary_map = {"Yes": 1, "No": 0}
    furnishing_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}

    payload = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": binary_map[mainroad],
        "guestroom": binary_map[guestroom],
        "basement": binary_map[basement],
        "hotwaterheating": binary_map[hotwaterheating],
        "airconditioning": binary_map[airconditioning],
        "parking": parking,
        "prefarea": binary_map[prefarea],
        "furnishingstatus": furnishing_map[furnishingstatus],
    }

    try:
        response = requests.post(api_url, json=payload, timeout=12)
        response.raise_for_status()
        result = response.json()

        probabilities = result.get("class_probabilities", [])
        predicted_label = result.get("predicted_class_label", "Unknown")

        top_probability = max(probabilities) if probabilities else 0
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Category", predicted_label)
        col2.metric("Confidence", f"{top_probability * 100:.2f}%")
        col3.metric("Input Area", f"{area:,} sq ft")

        st.subheader("Class Probability Distribution")
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(CLASS_NAMES, probabilities, color=["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"])
        ax.set_ylim([0, 1])
        ax.set_ylabel("Probability")
        ax.set_facecolor("#f8fbfd")

        for bar, value in zip(bars, probabilities):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        st.pyplot(fig)

        st.subheader("Confidence by Class")
        for name, score in zip(CLASS_NAMES, probabilities):
            st.write(f"**{name}**: {score * 100:.2f}%")
            st.progress(float(score))

    except requests.exceptions.RequestException as error:
        st.error("Unable to reach prediction API. Check FastAPI status and URL.")
        st.write(str(error))
    except Exception as error:
        st.error("Unexpected error occurred while generating prediction.")
        st.write(str(error))
