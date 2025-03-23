import streamlit as st
from streamlit_option_menu import option_menu
import os
import sys

# Add components directory to path if it exists
if os.path.exists("components"):
    sys.path.insert(0, "components")

# Import component modules
from components.dataset_manager import handle_dataset_upload
from components.model_selection import select_model
from components.hyperparameter_ui import configure_hyperparameters
from components.gemma_tuning import fine_tune_gemma
#from components.training_executor import display_training_metrics

# Set page configuration
st.set_page_config(
    page_title="Gemma Tune UI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better UI
def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        padding: 2rem 2rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    .sidebar .sidebar-content {
        background-color: #f1f5f9;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Initialize session state for data persistence between pages
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'jsonl_data' not in st.session_state:
    st.session_state.jsonl_data = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'hyperparams' not in st.session_state:
    st.session_state.hyperparams = None
if 'training_status' not in st.session_state:
    st.session_state.training_status = None
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = {}

# Sidebar with logo and navigation
with st.sidebar:
    st.image("assets/images/logo.png", width=150)
    st.title("Gemma AI TuneLab")
    
    selected_page = option_menu(
        menu_title="Navigation",
        options=[
            "Home",
            "Dataset Upload",
            "Data Preprocessing",
            "Model Selection",
            "Hyperparameter Tuning",
            "Training",
            "Visualization"
        ],
        icons=[
            "house",
            "cloud-upload",
            "tools",
            "cpu",
            "sliders",
            "play-circle",
            "graph-up"
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f1f5f9"},
            "icon": {"color": "#1E3A8A", "font-size": "16px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px", "--hover-color": "#e2e8f0"},
            "nav-link-selected": {"background-color": "#1E3A8A", "color": "white"},
        }
    )
    
    st.markdown("---")
    st.caption("© 2025 Gemma AI TuneLab")

# Main content area based on selected page
if selected_page == "Home":
    st.title("Welcome to Gemma AI TuneLab")
    
    st.markdown("""
    ### A No-Code Solution for Custom AI Model Training
    
    This application provides an intuitive interface for fine-tuning Gemma models without writing a single line of code.
    
    **Key Features:**
    - Upload and preprocess datasets in various formats (CSV, JSONL, text)
    - Configure hyperparameters through an interactive UI
    - Track real-time training progress with visualizations
    - Export fine-tuned models in multiple formats
    
    **Getting Started:**
    1. Upload your dataset using the 'Dataset Upload' section
    2. Preprocess and validate your data
    3. Select a Gemma model
    4. Configure your training hyperparameters
    5. Start training and monitor results
    
    Use the navigation menu on the left to get started!
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("#### 1. Data Management\nUpload, validate, and preprocess your training data")
    with col2:
        st.info("#### 2. Model Training\nSelect models and configure training parameters")
    with col3:
        st.info("#### 3. Deployment\nVisualize results and export your fine-tuned model")

elif selected_page == "Dataset Upload":
    st.title("Dataset Upload")
    st.markdown("Upload your training dataset in CSV, JSONL, or text format.")
    
    df, jsonl_data = handle_dataset_upload()
    
    if df is not None:
        st.session_state.dataset = df
        st.session_state.jsonl_data = jsonl_data
        st.success("Dataset uploaded successfully! Proceed to Data Preprocessing.")
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(5))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Records", len(df))
        with col2:
            st.metric("Number of Columns", len(df.columns))

elif selected_page == "Data Preprocessing":
    st.title("Data Preprocessing")
    
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset first.")
        st.button("Go to Dataset Upload", on_click=lambda: st.session_state.update({"_current_page": "Dataset Upload"}))
    else:
        st.subheader("Data Cleaning")
        
        col1, col2 = st.columns(2)
        with col1:
            drop_duplicates = st.checkbox("Remove Duplicate Entries", value=True)
            drop_na = st.checkbox("Remove Rows with Missing Values", value=True)
        with col2:
            normalize_text = st.checkbox("Normalize Text (lowercase, remove extra spaces)", value=True)
            remove_special_chars = st.checkbox("Remove Special Characters", value=False)
        
        if st.button("Apply Preprocessing"):
            # Placeholder for preprocessing logic
            cleaned_df = st.session_state.dataset.copy()
            
            if drop_duplicates:
                cleaned_df = cleaned_df.drop_duplicates()
            
            if drop_na:
                cleaned_df = cleaned_df.dropna()
                
            # In a real implementation, you would add more preprocessing steps
            
            st.session_state.dataset = cleaned_df
            st.success("Preprocessing applied successfully!")
            
            st.subheader("Processed Dataset Preview")
            st.dataframe(cleaned_df.head(5))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Records", len(st.session_state.dataset))
            with col2:
                st.metric("Processed Records", len(cleaned_df))

elif selected_page == "Model Selection":
    st.title("Model Selection")
    
    if st.session_state.dataset is None:
        st.warning("Please upload and preprocess a dataset first.")
    else:
        st.session_state.selected_model = select_model()
        
        if st.session_state.selected_model:
            st.success(f"Selected model: {st.session_state.selected_model}")
            
            st.subheader("Model Details")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Model Architecture**")
                st.info("Gemma - Google's lightweight and efficient LLM")
            with col2:
                st.markdown("**Model Size**")
                if "flash" in st.session_state.selected_model.lower():
                    st.info("Flash - Optimized for efficient fine-tuning")
                else:
                    st.info("Standard - Full model capabilities")

elif selected_page == "Hyperparameter Tuning":
    st.title("Hyperparameter Tuning")
    
    if st.session_state.selected_model is None:
        st.warning("Please select a model first.")
    else:
        st.session_state.hyperparams = configure_hyperparameters()
        
        if st.session_state.hyperparams:
            st.success("Hyperparameters configured successfully!")
            
            st.subheader("Configuration Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Training Parameters**")
                st.info(f"Learning Rate: {st.session_state.hyperparams['learning_rate']}")
                st.info(f"Batch Size: {st.session_state.hyperparams['batch_size']}")
                st.info(f"Epochs: {st.session_state.hyperparams['num_epochs']}")
            with col2:
                st.markdown("**Advanced Settings**")
                st.info(f"Optimizer: {st.session_state.hyperparams.get('optimizer', 'AdamW')}")
                st.info(f"Weight Decay: {st.session_state.hyperparams.get('weight_decay', 0.01)}")
                lora_rank = st.session_state.hyperparams.get('lora_rank', 8)
                st.info(f"LoRA Rank: {lora_rank}")

elif selected_page == "Training":
    st.title("Model Training")
    
    if (st.session_state.dataset is None or 
        st.session_state.selected_model is None or 
        st.session_state.hyperparams is None):
        st.warning("Please complete all previous steps before training.")
    else:
        st.subheader("Training Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Dataset**: {len(st.session_state.dataset)} records")
        with col2:
            st.info(f"**Model**: {st.session_state.selected_model}")
        with col3:
            st.info(f"**Epochs**: {st.session_state.hyperparams['num_epochs']}")
        
        st.subheader("Start Training")
        
        if st.button("Start Fine-Tuning"):
            # Placeholder for training progress
            st.session_state.training_status = "running"
            
            with st.spinner("Training in progress..."):
                # In a real implementation, this would be a non-blocking call
                training_data = list(zip(st.session_state.dataset["text"], 
                                         st.session_state.dataset["text"]))
                
                # Simulate training (in production, call fine_tune_gemma)
                import time
                progress_bar = st.progress(0)
                
                # Simulate epoch progress
                for i in range(st.session_state.hyperparams["num_epochs"]):
                    # Simulate batch progress
                    for j in range(10):  # 10 steps per epoch
                        time.sleep(0.2)  # Simulate computation
                        current_progress = (i * 10 + j + 1) / (st.session_state.hyperparams["num_epochs"] * 10)
                        progress_bar.progress(current_progress)
                        
                        # Store mock metrics
                        epoch = i + 1
                        if epoch not in st.session_state.training_metrics:
                            st.session_state.training_metrics[epoch] = []
                        
                        loss = 1.0 - (current_progress * 0.7)  # Mock decreasing loss
                        accuracy = current_progress * 0.8  # Mock increasing accuracy
                        
                        st.session_state.training_metrics[epoch].append({
                            'step': j + 1,
                            'loss': loss,
                            'accuracy': accuracy
                        })
                
                st.session_state.training_status = "completed"
            
            st.success("Training completed successfully!")
            st.balloons()
            
            st.subheader("Training Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Loss", f"{st.session_state.training_metrics[st.session_state.hyperparams['num_epochs']][-1]['loss']:.4f}")
            with col2:
                st.metric("Final Accuracy", f"{st.session_state.training_metrics[st.session_state.hyperparams['num_epochs']][-1]['accuracy']:.4f}")
            
            st.markdown("Proceed to the Visualization page to see detailed training metrics.")

elif selected_page == "Visualization":
    st.title("Training Visualization")
    
    if st.session_state.training_status != "completed":
        st.warning("No training data available. Please train a model first.")
    else:
        # Display training metrics using custom visualization component
        #display_training_metrics(st.session_state.training_metrics)
        
        st.subheader("Export Model")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Export as TensorFlow"):
                st.success("Model exported in TensorFlow format!")
        with col2:
            if st.button("Export as PyTorch"):
                st.success("Model exported in PyTorch format!")
        with col3:
            if st.button("Export as GGUF"):
                st.success("Model exported in GGUF format!")
        
        st.download_button(
            label="Download Model Card",
            data="Mock model card content",
            file_name="gemma_model_card.md",
            mime="text/markdown"
        )