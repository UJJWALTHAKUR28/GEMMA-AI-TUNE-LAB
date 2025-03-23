import streamlit as st
import pandas as pd
import numpy as np
import mimetypes
import json
import re
import random
import nltk
from nltk.corpus import wordnet
from transformers import AutoTokenizer
import plotly.express as px
import os
from io import StringIO

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download("wordnet", quiet=True)

# Load tokenizer (Using a publicly available model)
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

def handle_dataset_upload():
    """Handles dataset upload, conversion, preprocessing, and visualization"""
    st.subheader("Dataset Upload and Processing")
    
    # Create tabs for different upload methods
    tab1, tab2, tab3 = st.tabs(["File Upload", "Sample Data", "Text Input"])
    
    df = None
    jsonl_data = None
    
    with tab1:
        st.markdown("### Upload Your Dataset")
        st.markdown("Supported formats: CSV, JSONL, TXT")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Drag & Drop or Browse",
                type=["csv", "jsonl", "txt"],
                accept_multiple_files=False,
                help="Upload your training data in CSV, JSONL, or TXT format"
            )
        with col2:
            st.markdown("#### File Requirements")
            st.markdown("""
            - CSV: Must contain 'text' column
            - JSONL: Must have 'text' field
            - TXT: One example per line
            """)
        
        if uploaded_file is not None:
            df = process_uploaded_file(uploaded_file)
            
    with tab2:
        st.markdown("### Use Sample Dataset")
        st.markdown("Select a pre-configured sample dataset for testing")
        
        sample_option = st.selectbox(
            "Choose a sample dataset",
            ["None", "Question-Answer Pairs", "Text Classification", "Custom Instructions"],
            index=0
        )
        
        if sample_option != "None":
            df = load_sample_dataset(sample_option)
    
    with tab3:
        st.markdown("### Enter Text Directly")
        st.markdown("Enter examples one per line")
        
        text_input = st.text_area(
            "Enter your examples (one per line)",
            height=200,
            help="Each line will be treated as one training example"
        )
        
        if st.button("Create Dataset from Text"):
            if text_input.strip():
                df = pd.DataFrame({"text": text_input.strip().split('\n')})
                st.success("Dataset created from text input")
            else:
                st.warning("Please enter some text to create a dataset")
    
    # If we have a valid dataframe at this point
    if df is not None and not df.empty:
        with st.expander("Dataset Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
            st.info(f"Dataset contains {len(df)} records and {len(df.columns)} columns")
        
        # Show column mapping if needed
        if "text" not in df.columns and len(df.columns) > 0:
            st.warning("No 'text' column found. Please select which column to use as text data.")
            text_col = st.selectbox("Select text column", df.columns)
            df = df.rename(columns={text_col: "text"})
            st.success(f"Column '{text_col}' mapped to 'text'")
        
        # Preprocessing options
        st.markdown("---")
        st.markdown("### Data Preprocessing")
        
        with st.form("preprocessing_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                cleaning_options = st.multiselect(
                    "Text Cleaning Options",
                    ["Convert to lowercase", "Remove HTML tags", "Remove special characters", "Normalize whitespace", "Remove duplicates", "Drop empty rows"],
                    default=["Convert to lowercase", "Remove HTML tags", "Normalize whitespace", "Drop empty rows"]
                )
                
                max_length = st.slider(
                    "Maximum text length (tokens)",
                    min_value=32,
                    max_value=2048,
                    value=512,
                    step=32,
                    help="Texts longer than this will be truncated"
                )
            
            with col2:
                augmentation_options = st.multiselect(
                    "Text Augmentation Options",
                    ["Synonym replacement", "Back translation", "Random deletion", "None"],
                    default=["None"]
                )
                
                augmentation_percentage = st.slider(
                    "Augmentation percentage",
                    min_value=0,
                    max_value=100,
                    value=20,
                    step=5,
                    help="Percentage of examples to augment",
                    disabled="None" in augmentation_options
                )
            
            submitted = st.form_submit_button("Apply Preprocessing")
            
            if submitted:
                with st.spinner("Preprocessing dataset..."):
                    # Apply preprocessing based on selected options
                    df = preprocess_dataset(df, cleaning_options, max_length)
                    
                    # Apply augmentation if selected
                    if "None" not in augmentation_options:
                        df = augment_dataset(df, augmentation_options, augmentation_percentage/100)
                    
                    # Show tokenization preview
                    tokenize_and_visualize(df, max_length)
                    
                    # Convert to JSONL
                    jsonl_data = convert_to_jsonl(df)
                    
                    st.session_state.dataset = df
                    st.session_state.jsonl_data = jsonl_data
                    
                    st.success("Dataset successfully preprocessed and ready for training!")
        
        # Download processed dataset
        if jsonl_data is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Processed Dataset (JSONL)",
                    data=jsonl_data,
                    file_name="processed_dataset.jsonl",
                    mime="application/json"
                )
            with col2:
                st.download_button(
                    "Download Processed Dataset (CSV)",
                    data=df.to_csv(index=False),
                    file_name="processed_dataset.csv",
                    mime="text/csv"
                )
    
    return df, jsonl_data

def process_uploaded_file(uploaded_file):
    """Process the uploaded file based on its format"""
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == "jsonl" or file_extension == "json":
            content = uploaded_file.read().decode("utf-8")
            if file_extension == "json" and not content.startswith("{"):
                # Regular JSON file (not JSONL)
                df = pd.DataFrame(json.loads(content))
            else:
                # JSONL file
                df = pd.read_json(StringIO(content), lines=True)
        elif file_extension == "txt":
            content = uploaded_file.read().decode("utf-8")
            df = pd.DataFrame({"text": content.splitlines()})
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def load_sample_dataset(sample_option):
    """Load a pre-configured sample dataset"""
    if sample_option == "Question-Answer Pairs":
        df = pd.DataFrame({
            "text": [
                "What is machine learning?",
                "How does fine-tuning work?",
                "What are the best hyperparameters for LLMs?",
                "How can I optimize my model?",
                "What is the difference between LoRA and QLoRA?"
            ],
            "response": [
                "Machine learning is a subset of AI that enables systems to learn from data and improve over time without explicit programming.",
                "Fine-tuning adjusts pre-trained model weights on domain-specific data to specialize its capabilities.",
                "The best hyperparameters depend on your dataset and task, but common values include learning rates between 1e-5 and 5e-5, batch sizes of 4-16, and 2-5 epochs.",
                "You can optimize your model through quantization, pruning, knowledge distillation, and efficient fine-tuning techniques like LoRA.",
                "LoRA (Low-Rank Adaptation) uses low-rank matrices for efficient fine-tuning, while QLoRA adds quantization to reduce memory usage further."
            ]
        })
    elif sample_option == "Text Classification":
        df = pd.DataFrame({
            "text": [
                "I absolutely loved this product, it exceeded my expectations!",
                "The service was terrible and the staff was rude.",
                "It was okay, nothing special but did the job.",
                "I would not recommend this to anyone, complete waste of money.",
                "Best purchase I've made this year, definitely worth every penny!"
            ],
            "label": ["positive", "negative", "neutral", "negative", "positive"]
        })
    elif sample_option == "Custom Instructions":
        df = pd.DataFrame({
            "text": [
                "Summarize the following article in 3 bullet points.",
                "Translate the following English text to French.",
                "Write a creative story about a detective solving a mystery.",
                "Explain quantum computing to a 10-year-old.",
                "Create a meal plan for someone looking to lose weight."
            ]
        })
    else:
        return None
    
    st.success(f"Loaded sample dataset: {sample_option}")
    return df

def preprocess_dataset(df, cleaning_options, max_length):
    """Apply selected preprocessing options to the dataset"""
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Drop empty rows
    if "Drop empty rows" in cleaning_options:
        processed_df = processed_df.dropna(subset=["text"])
    
    # Remove duplicates
    if "Remove duplicates" in cleaning_options:
        original_count = len(processed_df)
        processed_df = processed_df.drop_duplicates(subset=["text"])
        removed = original_count - len(processed_df)
        if removed > 0:
            st.info(f"Removed {removed} duplicate entries")
    
    # Apply text cleaning based on selected options
    cleaning_funcs = []
    if "Convert to lowercase" in cleaning_options:
        cleaning_funcs.append(lambda x: x.lower())
    if "Remove HTML tags" in cleaning_options:
        cleaning_funcs.append(lambda x: re.sub(r"<[^>]+>", "", x))
    if "Remove special characters" in cleaning_options:
        cleaning_funcs.append(lambda x: re.sub(r"[^a-zA-Z0-9\s.,!?]", "", x))
    if "Normalize whitespace" in cleaning_options:
        cleaning_funcs.append(lambda x: re.sub(r"\s+", " ", x).strip())
    
    # Apply all selected cleaning functions
    if cleaning_funcs:
        for func in cleaning_funcs:
            processed_df["text"] = processed_df["text"].apply(func)
    
    # Truncate texts that are too long
    tokenizer = load_tokenizer()
    
    def truncate_text(text):
        tokens = tokenizer.encode(text)
        if len(tokens) > max_length:
            return tokenizer.decode(tokens[:max_length])
        return text
    
    processed_df["text"] = processed_df["text"].apply(truncate_text)
    
    return processed_df

def augment_dataset(df, augmentation_options, percentage):
    """Augment the dataset based on selected options"""
    # Make a copy of the original dataframe
    augmented_df = df.copy()
    
    # Calculate how many examples to augment
    num_to_augment = int(len(df) * percentage)
    
    if num_to_augment == 0:
        return augmented_df
    
    # Select random examples for augmentation
    indices_to_augment = np.random.choice(df.index, size=num_to_augment, replace=False)
    
    # Create new examples
    new_examples = []
    
    for idx in indices_to_augment:
        text = df.loc[idx, "text"]
        
        # Choose a random augmentation method
        if "Synonym replacement" in augmentation_options:
            augmented_text = synonym_replacement(text)
            new_row = df.loc[idx].copy()
            new_row["text"] = augmented_text
            new_examples.append(new_row)
            
        if "Back translation" in augmentation_options:
            augmented_text = mock_back_translation(text)
            new_row = df.loc[idx].copy()
            new_row["text"] = augmented_text
            new_examples.append(new_row)
            
        if "Random deletion" in augmentation_options:
            augmented_text = random_deletion(text)
            new_row = df.loc[idx].copy()
            new_row["text"] = augmented_text
            new_examples.append(new_row)
    
    # Add the new examples to the dataframe
    if new_examples:
        augmented_df = pd.concat([augmented_df, pd.DataFrame(new_examples)], ignore_index=True)
        st.info(f"Added {len(new_examples)} augmented examples. Dataset size increased from {len(df)} to {len(augmented_df)}")
    
    return augmented_df

def synonym_replacement(text, replace_count=2):
    """Replace words with their synonyms"""
    words = text.split()
    if len(words) <= 3:  # Don't replace if text is too short
        return text
        
    new_words = words.copy()
    replace_count = min(replace_count, len(words) // 3)  # Don't replace more than 1/3 of words
    
    # Find positions of words that have synonyms
    valid_positions = []
    for i, word in enumerate(words):
        if len(word) >= 4 and wordnet.synsets(word):  # Only consider words with at least 4 chars
            valid_positions.append(i)
    
    if not valid_positions:
        return text
    
    # Replace random words with synonyms
    for _ in range(min(replace_count, len(valid_positions))):
        if not valid_positions:
            break
            
        pos_idx = random.randint(0, len(valid_positions) - 1)
        word_idx = valid_positions[pos_idx]
        
        synonyms = []
        for syn in wordnet.synsets(words[word_idx]):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        
        if synonyms:
            # Select a synonym different from the original word
            valid_synonyms = [s for s in synonyms if s != words[word_idx]]
            if valid_synonyms:
                synonym = random.choice(valid_synonyms)
                new_words[word_idx] = synonym
        
        # Remove used position
        valid_positions.pop(pos_idx)
    
    return " ".join(new_words)

def mock_back_translation(text):
    """Mock back-translation (in a real system, this would use translation APIs)"""
    # This is just a simulation for demonstration
    words = text.split()
    # Shuffle word order slightly to simulate translation artifacts
    if len(words) > 3:
        cut_point = random.randint(1, len(words) - 2)
        segment = words[cut_point:cut_point+2]
        words[cut_point:cut_point+2] = reversed(segment)
    
    # Change some words to synonyms
    augmented = synonym_replacement(text, replace_count=3)
    
    return augmented

def random_deletion(text, p=0.1):
    """Randomly delete words from the text"""
    words = text.split()
    if len(words) <= 3:
        return text
    
    # Keep at least 80% of words
    keep_prob = max(0.8, 1 - p)
    
    new_words = []
    for word in words:
        # Keep stop words, punctuation, and short words
        if len(word) <= 2 or random.random() < keep_prob:
            new_words.append(word)
    
    # If we deleted too many words, keep the original
    if len(new_words) < len(words) * 0.5:
        return text
    
    return " ".join(new_words)

def tokenize_and_visualize(df, max_length):
    """Tokenize the dataset and visualize token distribution"""
    if df is None or "text" not in df.columns:
        return
    
    tokenizer = load_tokenizer()
    
    # Calculate token counts
    token_counts = [len(tokenizer.encode(text)) for text in df["text"].values]
    
    # Create a histogram of token counts
    fig = px.histogram(
        token_counts, 
        nbins=20, 
        title="Token Count Distribution",
        labels={"value": "Number of Tokens", "count": "Number of Examples"},
        color_discrete_sequence=["#1E3A8A"]
    )
    
    fig.add_vline(x=max_length, line_dash="dash", line_color="red", annotation_text=f"Max Length: {max_length}")
    
    # Add statistics
    stats = {
        "Min Tokens": min(token_counts),
        "Max Tokens": max(token_counts),
        "Mean Tokens": round(sum(token_counts) / len(token_counts), 1),
        "Median Tokens": sorted(token_counts)[len(token_counts) // 2],
        "Over Limit": sum(1 for count in token_counts if count > max_length)
    }
    
    # Display the histogram
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Min Tokens", stats["Min Tokens"])
    col2.metric("Max Tokens", stats["Max Tokens"])
    col3.metric("Mean Tokens", stats["Mean Tokens"])
    col4.metric("Median Tokens", stats["Median Tokens"])
    col5.metric("Examples > Max Length", stats["Over Limit"])
    
    # Warning if some examples are over the length limit
    if stats["Over Limit"] > 0:
        st.warning(f"{stats['Over Limit']} examples exceed the maximum token length and will be truncated.")
    
    return token_counts

def convert_to_jsonl(df):
    """Convert dataframe to JSONL format"""
    if df is None or df.empty:
        return None
    
    # Create a copy of the dataframe
    export_df = df.copy()
    
    # Ensure the dataframe has at least one row
    if len(export_df) == 0:
        return None
    
    # Save to JSONL
    jsonl_data = export_df.to_json(orient="records", lines=True)
    
    # Save to file
    with open("data/processed_dataset.jsonl", "w") as f:
        f.write(jsonl_data)
    
    return jsonl_data