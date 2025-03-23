import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_scheduler,
    DataCollatorForLanguageModeling
)
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm
from datetime import datetime

class TextDataset(Dataset):
    """Custom dataset for text data"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length",
            return_tensors="pt"
        )
        
        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = item["input_ids"].clone()
        
        return item

def execute_training(
    dataset=None,
    model_name="google/gemma-2b",
    num_epochs=5,
    batch_size=4,
    learning_rate=1e-4,
    weight_decay=0.01,
    use_lora=True,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.1,
    use_mixed_precision=True,
    max_length=512,
    eval_steps=50,
    save_steps=100,
    output_dir="./models"
):
    """
    Trains a model with advanced optimization techniques and displays live updates in Streamlit.
    
    Args:
        dataset: Dataset to train on (list of text samples)
        model_name: Base model to fine-tune
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        use_lora: Whether to use LoRA (Parameter-Efficient Fine-Tuning)
        lora_rank: Rank of LoRA layers
        lora_alpha: Alpha parameter for LoRA
        lora_dropout: Dropout rate for LoRA layers
        use_mixed_precision: Whether to use mixed precision training
        max_length: Maximum sequence length
        eval_steps: Number of steps between evaluations
        save_steps: Number of steps between model checkpoints
        output_dir: Directory to save model checkpoints
    """
    st.subheader("📊 Training Progress (Real-Time)")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Using device: {device}")
    
    # Set mixed precision if available
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    with st.spinner("Loading model..."):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_mixed_precision else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
    
    # Apply LoRA for parameter-efficient fine-tuning
    if use_lora:
        st.info("Applying LoRA for parameter-efficient fine-tuning")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    model.to(device)
    model.train()
    
    # Prepare dataset - if no dataset provided, use a small dummy dataset
    if dataset is None or len(dataset) == 0:
        st.warning("No dataset provided, using dummy data for demonstration")
        dataset = [
            "Hello, how are you today? I'm doing well, thank you for asking.",
            "I love working with AI models! They are so fascinating and useful.",
            "The weather is quite nice today. The sun is shining and it's warm outside.",
            "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."
        ]
    
    # Create dataset and split into train/validation
    full_dataset = TextDataset(dataset, tokenizer, max_length=max_length)
    
    # Split dataset: 90% train, 10% validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    st.info(f"Training on {train_size} samples, validating on {val_size} samples")
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling for causal LM
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=data_collator
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        collate_fn=data_collator
    )
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Placeholders for real-time updates
    progress_container = st.empty()
    metrics_container = st.container()
    
    with metrics_container:
        col1, col2 = st.columns(2)
        with col1:
            loss_chart_placeholder = st.empty()
        with col2:
            val_loss_chart_placeholder = st.empty()
    
    train_losses = []
    val_losses = []
    epochs = []
    
    # Function to update charts
    def update_charts():
        # Create loss chart
        fig1, ax1 = plt.subplots()
        ax1.plot(range(len(train_losses)), train_losses, label='Training Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        loss_chart_placeholder.pyplot(fig1)
        
        # Create validation loss chart if available
        if val_losses:
            fig2, ax2 = plt.subplots()
            ax2.plot(epochs, val_losses, 'o-', label='Validation Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Validation Loss')
            ax2.legend()
            val_loss_chart_placeholder.pyplot(fig2)
    
    # Function to evaluate model
    def evaluate():
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        model.train()
        return avg_val_loss
    
    # Main training loop
    global_step = 0
    best_val_loss = float('inf')
    
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        progress_text = f"Epoch {epoch}/{num_epochs}"
        progress_bar = progress_container.progress(0, text=progress_text)
        
        epoch_loss = 0
        steps_in_epoch = len(train_dataloader)
        
        for step, batch in enumerate(train_dataloader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Mixed precision training
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update tracking
            epoch_loss += loss.item()
            train_losses.append(loss.item())
            global_step += 1
            
            # Update progress bar
            progress_bar.progress(step / steps_in_epoch, text=f"{progress_text} - Loss: {loss.item():.4f}")
            
            # Evaluate and save checkpoint
            if global_step % eval_steps == 0:
                val_loss = evaluate()
                val_losses.append(val_loss)
                epochs.append(epoch - 1 + step / steps_in_epoch)
                st.info(f"Step {global_step} - Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_save_path = os.path.join(output_dir, "best_model")
                    model.save_pretrained(model_save_path)
                    tokenizer.save_pretrained(model_save_path)
                    st.success(f"New best model saved! Validation Loss: {val_loss:.4f}")
            
            # Regular checkpoints
            if global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(checkpoint_dir)
            
            # Update charts every few steps
            if global_step % 10 == 0 or step == steps_in_epoch:
                update_charts()
        
        # End of epoch
        avg_epoch_loss = epoch_loss / steps_in_epoch
        val_loss = evaluate()
        val_losses.append(val_loss)
        epochs.append(epoch)
        
        epoch_time = time.time() - epoch_start_time
        st.info(f"Epoch {epoch} completed in {epoch_time:.2f}s - Avg Loss: {avg_epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Save epoch checkpoint
        epoch_dir = os.path.join(output_dir, f"epoch-{epoch}")
        model.save_pretrained(epoch_dir)
        
        # Update charts
        update_charts()
    
    # Training complete
    training_time = time.time() - start_time
    st.success(f"🎉 Training Completed Successfully in {training_time:.2f}s!")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Generate training summary
    st.subheader("Training Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Training Loss", f"{train_losses[-1]:.4f}")
    with col2:
        st.metric("Best Validation Loss", f"{best_val_loss:.4f}")
    with col3:
        st.metric("Training Time", f"{training_time:.2f}s")
    
    return {
        "model_path": final_model_path,
        "best_model_path": os.path.join(output_dir, "best_model"),
        "training_loss": train_losses,
        "validation_loss": val_losses,
        "best_val_loss": best_val_loss,
        "training_time": training_time
    }

def generate_sample_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """
    Generate sample text from a trained model for demonstration
    """
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

def display_training_metrics(metrics, model_path=None, tokenizer_path=None):
    """
    Display training metrics and allow interactive text generation
    """
    st.subheader("Training Results")
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.line_chart(metrics.get("training_loss", []))
        st.write("Training Loss")
    
    with col2:
        if "validation_loss" in metrics and len(metrics["validation_loss"]) > 0:
            val_df = pd.DataFrame({
                "epoch": range(1, len(metrics["validation_loss"]) + 1),
                "validation_loss": metrics["validation_loss"]
            })
            st.line_chart(val_df.set_index("epoch"))
            st.write("Validation Loss")
    
    # Interactive text generation if model is available
    if model_path and tokenizer_path and os.path.exists(model_path):
        st.subheader("Interactive Text Generation")
        
        prompt = st.text_area("Enter a prompt:", "Once upon a time")
        
        col1, col2 = st.columns(2)
        with col1:
            max_length = st.slider("Max length:", 10, 500, 100)
        with col2:
            temperature = st.slider("Temperature:", 0.0, 1.0, 0.7)
        
        if st.button("Generate Text"):
            with st.spinner("Generating text..."):
                try:
                    # Load model for inference
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, 
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                    
                    generated_text = generate_sample_text(
                        model, tokenizer, prompt, max_length, temperature
                    )
                    
                    st.text_area("Generated text:", value=generated_text, height=200)
                except Exception as e:
                    st.error(f"Error generating text: {str(e)}")