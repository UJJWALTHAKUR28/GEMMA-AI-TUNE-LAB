import streamlit as st
import requests
import json
import os
import dotenv
from PIL import Image
import pandas as pd

# Load environment variables
dotenv.load_dotenv()
api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")

# Model information - expanded with more details
MODEL_OPTIONS = {
    "Gemma 1.5 Flash": {
        "id": "gemini-1.5-flash",
        "parameters": "1.5B",
        "description": "Lightweight model optimized for efficient fine-tuning and inference.",
        "context_window": "8K tokens",
        "best_for": ["Chatbots", "Content generation", "Summarization"],
        "memory_requirement": "Low",
        "image": "gemini_flash.png"  # Path to model image
    },
    "Gemma 1.5 Flash-8B": {
        "id": "gemini-1.5-flash-8b",
        "parameters": "8B",
        "description": "Medium-sized model with strong performance across various tasks.",
        "context_window": "16K tokens",
        "best_for": ["Complex reasoning", "Instruction following", "Creative writing"],
        "memory_requirement": "Medium",
        "image": "gemini_flash_8b.png"  # Path to model image
    },
    "Gemma 2B": {
        "id": "gemma-2b",
        "parameters": "2B",
        "description": "Compact model for lightweight applications and edge devices.",
        "context_window": "8K tokens",
        "best_for": ["Text classification", "Simple QA", "Resource-constrained environments"],
        "memory_requirement": "Low",
        "image": "gemma_2b.png"  # Path to model image
    },
    "Gemma 7B": {
        "id": "gemma-7b",
        "parameters": "7B",
        "description": "Versatile model with strong performance on a wide range of tasks.",
        "context_window": "8K tokens",
        "best_for": ["Content generation", "Text completion", "General-purpose applications"],
        "memory_requirement": "Medium",
        "image": "gemma_7b.png"  # Path to model image
    }
}

def select_model():
    """
    Enhanced model selection component with visual cards and detailed information.
    Includes fine-tuning configuration in the same interface.
    """
    st.subheader("Model Selection & Training Configuration")
    
    # Create tabs for different aspects of model selection
    tab1, tab2 = st.tabs(["Select Model", "Fine-Tuning Configuration"])
    
    with tab1:
        st.markdown("### Choose a Model for Fine-Tuning")
        st.markdown("Select the Gemma model that best suits your application needs:")
        
        # Create a grid of model cards
        col1, col2 = st.columns(2)
        
        selected_model_name = None
        
        # First row
        with col1:
            if create_model_card("Gemma 1.5 Flash", MODEL_OPTIONS["Gemma 1.5 Flash"]):
                selected_model_name = "Gemma 1.5 Flash"
                
        with col2:
            if create_model_card("Gemma 1.5 Flash-8B", MODEL_OPTIONS["Gemma 1.5 Flash-8B"]):
                selected_model_name = "Gemma 1.5 Flash-8B"
        
        # Second row
        col3, col4 = st.columns(2)
        
        with col3:
            if create_model_card("Gemma 2B", MODEL_OPTIONS["Gemma 2B"]):
                selected_model_name = "Gemma 2B"
                
        with col4:
            if create_model_card("Gemma 7B", MODEL_OPTIONS["Gemma 7B"]):
                selected_model_name = "Gemma 7B"
        
        # Store selection in session state
        if selected_model_name:
            st.session_state.selected_model_name = selected_model_name
            st.session_state.selected_model = MODEL_OPTIONS[selected_model_name]["id"]
    
    with tab2:
        if "selected_model_name" not in st.session_state:
            st.info("Please select a model in the 'Select Model' tab first.")
            return None
        
        st.markdown(f"### Fine-Tuning Configuration for {st.session_state.selected_model_name}")
        
        # Create columns for different configuration categories
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Training Parameters")
            
            # Learning rate with logarithmic slider
            learning_rate = st.slider(
                "Learning Rate",
                min_value=0.000001, 
                max_value=0.001,
                value=0.0001,
                format="%.6f",
                help="Controls how quickly the model adapts to the training data."
            )
            
            # Batch size
            batch_size = st.select_slider(
                "Batch Size",
                options=[1, 2, 4, 8, 16, 32],
                value=4,
                help="Number of examples processed together in each training step."
            )
            
            # Epochs
            epochs = st.slider(
                "Number of Epochs",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of complete passes through the training dataset."
            )
        
        with col2:
            st.markdown("#### Advanced Configuration")
            
            # Weight decay
            weight_decay = st.slider(
                "Weight Decay",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                format="%.2f",
                help="Regularization parameter to prevent overfitting."
            )
            
            # LoRA settings
            use_lora = st.checkbox(
                "Use LoRA (Low-Rank Adaptation)",
                value=True,
                help="Efficient fine-tuning technique that reduces memory requirements."
            )
            
            if use_lora:
                lora_rank = st.slider(
                    "LoRA Rank",
                    min_value=4,
                    max_value=64,
                    value=8,
                    help="Rank of LoRA adaptation matrices. Lower values use less memory."
                )
            else:
                lora_rank = None
            
            # Optimizer
            optimizer = st.selectbox(
                "Optimizer",
                options=["AdamW", "Adam", "SGD"],
                index=0,
                help="Algorithm used to update the model weights during training."
            )
        
        # Store configuration in session state
        if st.button("Save Configuration", type="primary"):
            st.session_state.hyperparams = {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": epochs,
                "weight_decay": weight_decay,
                "use_lora": use_lora,
                "lora_rank": lora_rank,
                "optimizer": optimizer
            }
            
            st.success("✓ Configuration saved successfully!")
            
            # Display configuration summary
            st.markdown("### Configuration Summary")
            
            # Create a DataFrame for better visualization
            config_data = {
                "Parameter": ["Model", "Learning Rate", "Batch Size", "Epochs", 
                             "Weight Decay", "LoRA", "LoRA Rank", "Optimizer"],
                "Value": [
                    st.session_state.selected_model_name,
                    f"{learning_rate:.6f}",
                    str(batch_size),
                    str(epochs),
                    f"{weight_decay:.2f}",
                    "Enabled" if use_lora else "Disabled",
                    str(lora_rank) if use_lora else "N/A",
                    optimizer
                ]
            }
            
            config_df = pd.DataFrame(config_data)
            st.dataframe(config_df, hide_index=True, use_container_width=True)
    
    # Return the selected model ID if the configuration is complete
    if "selected_model" in st.session_state and "hyperparams" in st.session_state:
        return st.session_state.selected_model
    else:
        return None

def create_model_card(name, model_info):
    """Creates an interactive card for a model with details and selection button."""
    
    # Create a container with a border
    with st.container():
        st.markdown(
            f"""
            <div style="
                border: 1px solid #e0e0e0; 
                border-radius: 10px; 
                padding: 15px; 
                margin-bottom: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                background-color: {'#f0f7ff' if 'selected_model_name' in st.session_state and st.session_state.selected_model_name == name else 'white'};
                ">
                <h3 style="margin-top: 0; color: #1E3A8A;">{name}</h3>
                <p><strong>Parameters:</strong> {model_info['parameters']}</p>
                <p><strong>Context Window:</strong> {model_info['context_window']}</p>
                <p>{model_info['description']}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Creating two columns for details and button
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("**Best for:**")
            for use_case in model_info['best_for']:
                st.markdown(f"- {use_case}")
        
        with col2:
            # Select button
            is_selected = st.button(
                "Select" if 'selected_model_name' not in st.session_state or st.session_state.selected_model_name != name else "Selected ✓", 
                key=f"select_{name}",
                type="primary" if 'selected_model_name' not in st.session_state or st.session_state.selected_model_name != name else "secondary"
            )
            
    return is_selected

def fine_tune_gemma(model_id, training_data, hyperparams=None):
    """
    Fine-tunes Gemma models using Google AI Studio API
    
    Parameters:
    - model_id: The ID of the model to fine-tune
    - training_data: List of (input, output) tuples for training
    - hyperparams: Dictionary of hyperparameters
    
    Returns:
    - job_id: The ID of the fine-tuning job
    """
    st.subheader("Model Fine-Tuning")

    if not api_key:
        st.error("API key is missing. Please set it in the `.env` file.")
        st.info("You can get an API key from Google AI Studio.")
        return None

    api_url = "https://generativelanguage.googleapis.com/v1/tune"
    
    # Use default hyperparameters if none provided
    if not hyperparams:
        hyperparams = {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "use_lora": True,
            "lora_rank": 8,
            "optimizer": "AdamW"
        }

    # Convert dataset into Google AI's expected format
    training_dataset = [
        {"input": example[0], "output": example[1]} for example in training_data
    ]

    # Create the request payload
    payload = {
        "base_model": model_id,
        "training_data": training_dataset,
        "tuning_method": "lora" if hyperparams.get("use_lora", True) else "full",
        "config": {
            "epoch_count": hyperparams.get("num_epochs", 3),
            "batch_size": hyperparams.get("batch_size", 4),
            "learning_rate": hyperparams.get("learning_rate", 0.0001),
            "weight_decay": hyperparams.get("weight_decay", 0.01),
            "optimizer": hyperparams.get("optimizer", "AdamW"),
            "tuned_model_display_name": f"Fine-Tuned-{model_id}"
        }
    }
    
    # Add LoRA specific parameters if LoRA is enabled
    if hyperparams.get("use_lora", True):
        payload["config"]["lora_rank"] = hyperparams.get("lora_rank", 8)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Show API request information in an expandable section
    with st.expander("View API Request Details"):
        st.json({
            "url": api_url,
            "payload": payload
        })

    # Make request to Google AI Studio API
    try:
        # Create a progress indicator
        progress_text = "Submitting fine-tuning job..."
        progress_bar = st.progress(0)
        
        for i in range(100):
            # Update progress bar
            progress_bar.progress(i + 1)
            
            # When we reach 100%, make the API call
            if i == 99:
                response = requests.post(api_url, headers=headers, json=payload)
                break
                
            # Small delay to show progress
            import time
            time.sleep(0.01)

        # Handle Response
        if response.status_code == 200:
            try:
                response_data = response.json()
                job_id = response_data.get("job_id", "unknown")
                
                st.success(f"Fine-tuning job started successfully!")
                
                # Display job information
                st.markdown("### Job Information")
                st.markdown(f"**Job ID:** {job_id}")
                st.markdown(f"**Model:** {model_id}")
                st.markdown(f"**Status:** Running")
                
                # Provide next steps
                st.info("You can check the job status in Google AI Studio or use the 'Training' tab to monitor progress.")
                
                return job_id
                
            except json.JSONDecodeError:
                st.warning("Fine-tuning started, but response could not be parsed as JSON.")
                st.code(response.text)
                return None
        else:
            try:
                error_data = response.json()
                st.error(f"Failed to start fine-tuning: {error_data}")
            except json.JSONDecodeError:
                st.error(f"Failed to start fine-tuning. API returned: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None