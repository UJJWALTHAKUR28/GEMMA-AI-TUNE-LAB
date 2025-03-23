import streamlit as st
import requests
import json
import os
import dotenv

# Load API key directly (to prevent circular import)
dotenv.load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")

def fine_tune_gemma(model_name, training_data, num_epochs=5, batch_size=4, learning_rate=0.001):
    """Fine-tunes Gemini 1.5 Flash models using Google AI Studio API"""

    st.subheader("🔧 Fine-Tuning Gemini Model")

    if not api_key:
        st.error("🚨 API key is missing. Please set it in the `.env` file.")
        return

    api_url = "https://generativelanguage.googleapis.com/v1/tune"  # Update with actual Google AI Studio endpoint

    # Convert dataset into Google AI's expected format
    training_dataset = [
        {"input": example[0], "output": example[1]} for example in training_data
    ]

    # Request payload
    payload = {
        "base_model": model_name,
        "training_data": training_dataset,
        "config": {
            "epoch_count": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "tuned_model_display_name": "Fine-Tuned Gemini"
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Make request to Google AI Studio API
    try:
        response = requests.post(api_url, headers=headers, json=payload)

        # Handle Response
        if response.status_code == 200:
            try:
                response_data = response.json()
                st.success(f"✅ Fine-tuning started for {model_name}! Check Google AI Studio for progress.")
            except json.JSONDecodeError:
                st.warning("⚠️ Fine-tuning started, but response could not be parsed as JSON.")
        else:
            try:
                error_data = response.json()
                st.error(f"🚨 Failed to start fine-tuning: {error_data}")
            except json.JSONDecodeError:
                st.error(f"🚨 Failed to start fine-tuning. API returned: {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Network error: {str(e)}")
