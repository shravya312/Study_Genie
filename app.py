import streamlit as st
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import re

# Set page config
st.set_page_config(
    page_title="Study Genie - Sentiment Analysis & Text Summarization",
    page_icon="📚",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        height: 200px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📚 Study Genie")
st.markdown("""
    Welcome to Study Genie! This app helps you analyze text sentiment and generate summaries.
    Simply enter your text below and choose the feature you want to use.
""")

# Initialize models
@st.cache_resource
def load_models():
    # Load sentiment analysis model
    sentiment_model = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
    
    # Load summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    
    return sentiment_model, summarizer

# Load models
try:
    sentiment_model, summarizer = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Create tabs for different features
tab1, tab2 = st.tabs(["Sentiment Analysis", "Text Summarization"])

# Sentiment Analysis Tab
with tab1:
    st.header("Sentiment Analysis")
    text_input = st.text_area("Enter text to analyze:", height=200)
    
    if st.button("Analyze Sentiment"):
        if text_input:
            with st.spinner("Analyzing sentiment..."):
                try:
                    result = sentiment_model(text_input)
                    sentiment = result[0]['label']
                    score = result[0]['score']
                    
                    # Display results
                    st.subheader("Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sentiment", sentiment)
                    with col2:
                        st.metric("Confidence", f"{score:.2%}")
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")

# Text Summarization Tab
with tab2:
    st.header("Text Summarization")
    text_input = st.text_area("Enter text to summarize:", height=200)
    
    col1, col2 = st.columns(2)
    with col1:
        min_length = st.slider("Minimum summary length", 30, 100, 50)
    with col2:
        max_length = st.slider("Maximum summary length", 100, 300, 150)
    
    if st.button("Generate Summary"):
        if text_input:
            with st.spinner("Generating summary..."):
                try:
                    summary = summarizer(text_input, 
                                       max_length=max_length, 
                                       min_length=min_length, 
                                       do_sample=True)
                    
                    st.subheader("Summary")
                    st.write(summary[0]['summary_text'])
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
        else:
            st.warning("Please enter some text to summarize.")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ by Study Genie") 