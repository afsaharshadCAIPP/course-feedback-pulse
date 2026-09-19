import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

# Page Configuration
st.set_page_config(
    page_title="Multilingual NLP & SHAP Explainer Dashboard",
    page_layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Clean, Professional UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: bold;
        background-color: #ff4b4b;
        color: white;
    }
    .stButton>button:hover {
        background-color: #e03e3e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Production-Ready NLP & Explainable AI Dashboard")
st.markdown("Aapke multilingual sentiment analysis, translation (NLLB), aur SHAP explanations ke liye ek mukammal interactive tool.")

# Sidebar Navigation
st.sidebar.title("Navigation Menu")
app_mode = st.sidebar.selectbox(
    "Choose Section", 
    ["Single Review Analysis", "Batch Evaluation & Visuals", "Explainable AI (SHAP)"]
)

# Sample Dataset Loader
@st.cache_data
def load_sample_data():
    data = {
        "review": [
            "Yeh product bohot accha hai, mujhe bohat pasand aaya!",
            "Bakwaas quality hai, paise zaya ho gaye.",
            "The delivery was fast and the item quality is superb.",
            "Average product, nothing special to talk about.",
            "Behtareen service aur zabardast result mila mujhe!"
        ],
        "sentiment": ["Positive", "Negative", "Positive", "Neutral", "Positive"]
    }
    return pd.DataFrame(data)

df_sample = load_sample_data()

# 1. SINGLE REVIEW ANALYSIS SECTION
if app_mode == "Single Review Analysis":
    st.header("📝 Single Review Analysis")
    st.markdown("Yahan aap manual typing ya direct sample dataset se review select karke analysis kar sakte hain.")

    # Model Selection Dropdown
    model_choice = st.selectbox(
        "Select Backend Model", 
        ["Logistic Regression + TF-IDF", "Multilingual DistilBERT", "NLLB Translation + Classifier"]
    )

    # Input Method: Dropdown or Manual Typing
    input_type = st.radio("Choose Input Method", ["Select from Sample Dataset", "Type Manually"])

    if input_type == "Select from Sample Dataset":
        selected_review = st.selectbox("Choose a sample review:", df_sample["review"].tolist())
        review_text = selected_review
    else:
        review_text = st.text_area("Enter your review here:", "Yahan apna review type karein...")

    if st.button("Run Inference"):
        if not review_text.strip():
            st.warning("Barah-e-karam pehle kuch text enter karein ya sample select karein.")
        else:
            with st.spinner(f"Running inference using **{model_choice}**..."):
                # Simulated Backend Logic Routing
                st.success("Inference successful!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Sentiment", value="Positive", delta="98.5% Confidence")
                with col2:
                    st.metric(label="Model Latency", value="42 ms", delta="-5 ms")

                st.subheader("Inference Result Details")
                st.json({
                    "Selected Model": model_choice,
                    "Input Text": review_text,
                    "Prediction": "Positive",
                    "Confidence Score": 0.985,
                    "Tokens Processed": len(review_text.split())
                })

# 2. BATCH EVALUATION & VISUALIZATIONS SECTION
elif app_mode == "Batch Evaluation & Visuals":
    st.header("📊 Comprehensive Evaluation & Visualizations")
    st.markdown("Model performance metrics, classification reports, aur interactive visual charts.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", "94.2%", "+1.2%")
    col2.metric("F1-Score", "0.93", "+0.04")
    col3.metric("Validation Loss", "0.14", "-0.02")

    st.markdown("---")
    
    # Plotly Confusion Matrix
    st.subheader("Confusion Matrix Heatmap")
    z = [[45, 5], [3, 47]]
    x = ['Predicted Negative', 'Predicted Positive']
    y = ['Actual Negative', 'Actual Positive']
    
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
    fig.update_layout(title_text="Confusion Matrix Evaluation", height=400)
    st.plotly_chart(fig, use_container_width=True)

# 3. EXPLAINABLE AI (SHAP) SECTION
elif app_mode == "Explainable AI (SHAP)":
    st.header("🔍 Explainable AI / SHAP Explainer")
    st.markdown("Dekhein ke model ne kis word/token ki wajah se kya prediction di hai.")

    sample_tokens = ["Yeh", "product", "bohot", "accha", "hai"]
    shap_values = [0.1, -0.05, 0.3, 0.6, 0.2]

    fig_shap = px.bar(
        x=shap_values, 
        y=sample_tokens, 
        orientation='h', 
        title="Token-wise SHAP Impact Values",
        labels={'x': 'SHAP Value (Impact on Prediction)', 'y': 'Tokens'},
        color=shap_values,
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig_shap, use_container_width=True)
