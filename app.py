import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from eda import generate_wordcloud, plot_label_distribution, run_eda
from fetch_trending import fetch_trending_headlines
from clickbait_detector import predict_clickbait, train_model
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and tokenizer
@st.cache_resource
def load_model():
    try:
        tokenizer = DistilBertTokenizer.from_pretrained("./clickbait_detector_model")
        model = DistilBertForSequenceClassification.from_pretrained("./clickbait_detector_model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None, None

# Streamlit app
st.title("Clickbait Detector")
st.markdown("Built by Chaw Thiri San")

# Ensure model is trained
if not os.path.exists("./clickbait_detector_model"):
    st.warning("Model not found. Training a new model...")
    if train_model(force_training=True):
        st.success("Model trained successfully!")
    else:
        st.error("Failed to train model. Check logs for details.")

tokenizer, model, device = load_model()
if tokenizer is None:
    st.error("Model not loaded. Please train the model first.")
    st.stop()

# Single headline prediction
st.header("Predict a Headline")
user_input = st.text_input("Enter a headline:", "You won't believe what happened next!")
if st.button("Predict"):
    pred, conf = predict_clickbait(user_input)
    if pred:
        st.write(f"**Prediction**: {pred} (Confidence: {conf:.2%})")
    else:
        st.error("Prediction failed. Please try again.")

# Trending headlines
st.header("Trending Headlines")
api_key = st.text_input("Enter News API key (optional):", type="password")
if st.button("Fetch Trending Headlines"):
    trending_df = fetch_trending_headlines(api_key)
    if trending_df.empty:
        st.error("Failed to fetch trending headlines. Check API key or model availability.")
    else:
        # Store corrections in session state
        if "corrections" not in st.session_state:
            st.session_state.corrections = {}
        
        for i, row in trending_df.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Headline**: {row['headline']} -> **Prediction**: {row['label']} (Confidence: {row['confidence']:.2%})")
            with col2:
                correction = st.selectbox(
                    "Correct label?", 
                    ["No change", "Clickbait", "Non-Clickbait"], 
                    key=f"correct_{i}"
                )
                if correction != "No change":
                    st.session_state.corrections[row["headline"]] = correction
        
        # Save corrections
        if st.button("Save Corrections"):
            corrections_df = pd.DataFrame(
                [{"headline": k, "label": v} for k, v in st.session_state.corrections.items()]
            )
            if not corrections_df.empty:
                corrections_df.to_csv("corrected_headlines.csv", index=False)
                st.success("Corrections saved to corrected_headlines.csv")
                # Update dataset
                from update_dataset import update_dataset
                if update_dataset():
                    st.success("Dataset updated with corrections")
                else:
                    st.error("Failed to update dataset")
            else:
                st.info("No corrections to save")

# EDA section
st.header("Dataset Insights")
if st.button("Generate EDA Visuals"):
    if run_eda():
        st.success("EDA visuals generated successfully!")
    else:
        st.error("Failed to generate EDA visuals. Check logs for details.")

# Display EDA
try:
    df = pd.read_csv("headlines_dataset.csv")
    st.subheader("Label Distribution")
    fig = plot_label_distribution(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to generate label distribution chart")
    
    st.subheader("Word Clouds")
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("clickbait_wordcloud.png"):
            st.image("clickbait_wordcloud.png", caption="Clickbait Headlines")
        else:
            st.warning("Clickbait word cloud not found")
    with col2:
        if os.path.exists("non_clickbait_wordcloud.png"):
            st.image("non_clickbait_wordcloud.png", caption="Non-Clickbait Headlines")
        else:
            st.warning("Non-Clickbait word cloud not found")
except FileNotFoundError:
    st.error("Dataset not found. Please ensure headlines_dataset.csv exists.")

