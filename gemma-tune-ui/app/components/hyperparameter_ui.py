import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

def configure_hyperparameters():
    """
    Advanced hyperparameter configuration interface with visualization of impact
    and recommendation features.
    """
    st.subheader("⚙️ Advanced Hyperparameter Configuration")
    
    # Create tabs for different aspects of hyperparameter tuning
    basic_tab, advanced_tab, visualization_tab, presets_tab = st.tabs([
        "Basic Parameters", 
        "Advanced Parameters", 
        "Impact Visualization",
        "Presets & Recommendations"
    ])
    
    # Initialize hyperparameters dictionary
    if "temp_hyperparams" not in st.session_state:
        st.session_state.temp_hyperparams = {
            "learning_rate": 0.0001,
            "batch_size": 4,
            "num_epochs": 3,
            "lora_rank": 8,
            "weight_decay": 0.01,
            "optimizer": "AdamW",
            "use_lora": True,
            "dropout": 0.1,
            "warmup_steps": 100,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0
        }
    
    # BASIC PARAMETERS TAB
    with basic_tab:
        col1, col2 = st.columns(2)
        
        with col1:
            # Learning rate with logarithmic slider and scientific notation
            st.markdown("#### Learning Rate")
            lr_exp = st.slider(
                "Select exponent (10^x)",
                min_value=-6,
                max_value=-3,
                value=-4,
                help="Controls how quickly the model adapts to the training data."
            )
            lr_mantissa = st.slider(
                "Fine-tune learning rate",
                min_value=1.0,
                max_value=9.9,
                value=1.0,
                step=0.1,
                help="Fine tune the learning rate multiplier."
            )
            st.session_state.temp_hyperparams["learning_rate"] = lr_mantissa * (10 ** lr_exp)
            st.write(f"Learning rate: {st.session_state.temp_hyperparams['learning_rate']:.8f}")
            
            # Batch size with memory estimation
            st.markdown("#### Batch Size")
            selected_batch_size = st.select_slider(
                "Select batch size",
                options=[1, 2, 4, 8, 16, 32],
                value=st.session_state.temp_hyperparams["batch_size"],
                help="Number of examples processed together in each training step."
            )
            st.session_state.temp_hyperparams["batch_size"] = selected_batch_size
            
            # Show estimated memory usage based on batch size
            memory_estimates = {
                1: "~4GB", 
                2: "~6GB", 
                4: "~8GB", 
                8: "~12GB", 
                16: "~18GB", 
                32: "~28GB"
            }
            st.info(f"Estimated GPU memory required: {memory_estimates[selected_batch_size]}")
        
        with col2:
            # Number of epochs with time estimate
            st.markdown("#### Training Duration")
            num_epochs = st.slider(
                "Number of Epochs",
                min_value=1,
                max_value=10,
                value=st.session_state.temp_hyperparams["num_epochs"],
                help="Number of complete passes through the training dataset."
            )
            st.session_state.temp_hyperparams["num_epochs"] = num_epochs
            
            # Show estimated training time based on epochs and batch size
            dataset_size = 10000  # Placeholder - could get from session state in real app
            time_per_batch = 0.5  # seconds, placeholder
            steps_per_epoch = dataset_size / selected_batch_size
            total_training_time = steps_per_epoch * num_epochs * time_per_batch
            hours = int(total_training_time / 3600)
            minutes = int((total_training_time % 3600) / 60)
            
            st.info(f"Estimated training time: {hours}h {minutes}m")
            
            # Optimizer selection
            st.markdown("#### Optimization Method")
            optimizer = st.selectbox(
                "Optimizer",
                options=["AdamW", "Adam", "SGD", "Adafactor", "Lion"],
                index=0,
                help="Algorithm used to update the model weights during training."
            )
            st.session_state.temp_hyperparams["optimizer"] = optimizer
            
            # Display optimizer info
            optimizer_info = {
                "AdamW": "Recommended for most fine-tuning tasks. Includes weight decay.",
                "Adam": "Standard optimizer without decoupled weight decay.",
                "SGD": "Simple but may require more tuning.",
                "Adafactor": "Memory-efficient alternative to Adam.",
                "Lion": "New optimizer that often allows larger learning rates."
            }
            st.caption(optimizer_info[optimizer])
    
    # ADVANCED PARAMETERS TAB
    with advanced_tab:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Regularization Parameters")
            
            # Weight decay
            weight_decay = st.slider(
                "Weight Decay",
                min_value=0.0,
                max_value=0.1,
                value=st.session_state.temp_hyperparams["weight_decay"],
                format="%.3f",
                help="Regularization parameter to prevent overfitting."
            )
            st.session_state.temp_hyperparams["weight_decay"] = weight_decay
            
            # Dropout
            dropout = st.slider(
                "Dropout",
                min_value=0.0,
                max_value=0.5,
                value=st.session_state.temp_hyperparams["dropout"],
                format="%.2f",
                help="Randomly drops units during training to prevent overfitting."
            )
            st.session_state.temp_hyperparams["dropout"] = dropout
            
            # Maximum gradient norm
            max_grad_norm = st.slider(
                "Max Gradient Norm",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.temp_hyperparams["max_grad_norm"],
                format="%.1f",
                help="Clips gradients to prevent exploding gradients."
            )
            st.session_state.temp_hyperparams["max_grad_norm"] = max_grad_norm
        
        with col2:
            st.markdown("#### Training Optimizations")
            
            # LoRA settings
            use_lora = st.checkbox(
                "Use LoRA (Low-Rank Adaptation)",
                value=st.session_state.temp_hyperparams["use_lora"],
                help="Efficient fine-tuning technique that reduces memory requirements."
            )
            st.session_state.temp_hyperparams["use_lora"] = use_lora
            
            if use_lora:
                lora_rank = st.slider(
                    "LoRA Rank",
                    min_value=1,
                    max_value=64,
                    value=st.session_state.temp_hyperparams["lora_rank"],
                    help="Rank of LoRA adaptation matrices. Lower values use less memory."
                )
                st.session_state.temp_hyperparams["lora_rank"] = lora_rank
                
                # Add a visual guide for LoRA rank trade-off
                st.markdown("##### LoRA Rank Trade-off")
                col1, col2, col3 = st.columns(3)
                col1.metric("Memory", "Lower ↓" if lora_rank < 16 else "Higher ↑")
                col2.metric("Training Speed", "Faster ↑" if lora_rank < 16 else "Slower ↓")
                col3.metric("Model Quality", "Lower ↓" if lora_rank < 16 else "Higher ↑")
            
            # Warmup steps
            warmup_steps = st.slider(
                "Warmup Steps",
                min_value=0,
                max_value=500,
                value=st.session_state.temp_hyperparams["warmup_steps"],
                help="Number of steps for linear learning rate warmup."
            )
            st.session_state.temp_hyperparams["warmup_steps"] = warmup_steps
            
            # Gradient accumulation
            grad_accum_steps = st.slider(
                "Gradient Accumulation Steps",
                min_value=1,
                max_value=8,
                value=st.session_state.temp_hyperparams["gradient_accumulation_steps"],
                help="Accumulate gradients over multiple steps (simulates larger batch size)."
            )
            st.session_state.temp_hyperparams["gradient_accumulation_steps"] = grad_accum_steps
            
            # Show effective batch size
            effective_batch = selected_batch_size * grad_accum_steps
            st.info(f"Effective batch size: {effective_batch}")
    
    # VISUALIZATION TAB
    with visualization_tab:
        st.markdown("### Training Dynamics Visualization")
        st.caption("Visualize the potential impact of your hyperparameter choices on training")
        
        # Select visualization type
        viz_type = st.selectbox(
            "Select visualization",
            ["Learning Rate Schedule", "Training Loss Projection", "Memory vs. Performance Trade-off"]
        )
        
        if viz_type == "Learning Rate Schedule":
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Generate data for learning rate schedule with warmup
            steps = np.arange(0, 1000)
            lr = np.zeros_like(steps, dtype=float)
            
            # Warmup phase
            warmup = st.session_state.temp_hyperparams["warmup_steps"]
            for i in range(warmup):
                lr[i] = (i / warmup) * st.session_state.temp_hyperparams["learning_rate"]
            
            # Constant or decay phase
            decay_type = st.radio("Decay Type", ["Constant", "Linear Decay", "Cosine Decay"])
            
            if decay_type == "Constant":
                lr[warmup:] = st.session_state.temp_hyperparams["learning_rate"]
            elif decay_type == "Linear Decay":
                for i in range(warmup, 1000):
                    progress = (i - warmup) / (1000 - warmup)
                    lr[i] = st.session_state.temp_hyperparams["learning_rate"] * (1 - progress)
            else:  # Cosine decay
                for i in range(warmup, 1000):
                    progress = (i - warmup) / (1000 - warmup)
                    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                    lr[i] = st.session_state.temp_hyperparams["learning_rate"] * cosine_decay
            
            ax.plot(steps, lr)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Convert the Matplotlib figure to a format compatible with Streamlit
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            st.image(buf)
            
        elif viz_type == "Training Loss Projection":
            # Generate mock training loss curves based on hyperparameters
            fig, ax = plt.subplots(figsize=(10, 5))
            
            epochs = np.arange(1, st.session_state.temp_hyperparams["num_epochs"] + 1)
            steps_per_epoch = 100
            
            # Generate baseline loss curve
            baseline_loss = 2.0 * np.exp(-0.5 * epochs) + 0.5
            
            # Modify based on learning rate (too high = unstable, too low = slow)
            lr_factor = 1.0
            if st.session_state.temp_hyperparams["learning_rate"] > 0.001:
                lr_factor = 1.2  # More unstable
            elif st.session_state.temp_hyperparams["learning_rate"] < 0.0001:
                lr_factor = 0.8  # Slower convergence
                
            # Add regularization effect (weight decay)
            wd_factor = 1.0
            if st.session_state.temp_hyperparams["weight_decay"] > 0.02:
                wd_factor = 0.9  # Better generalization
                
            # Create projected loss curves
            training_loss = baseline_loss * lr_factor
            validation_loss = baseline_loss * lr_factor * wd_factor + 0.2
            
            # Add some noise
            np.random.seed(42)
            training_loss += np.random.normal(0, 0.05, len(epochs))
            validation_loss += np.random.normal(0, 0.08, len(epochs))
            
            ax.plot(epochs, training_loss, 'b-', label='Training Loss')
            ax.plot(epochs, validation_loss, 'r-', label='Validation Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_title('Projected Training Dynamics')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Convert the Matplotlib figure to a format compatible with Streamlit
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            st.image(buf)
            
            # Add interpretation
            st.markdown("### Interpretation")
            if validation_loss[-1] > training_loss[-1] + 0.3:
                st.warning("⚠️ Potential overfitting detected. Consider increasing weight decay or using dropout.")
            elif validation_loss[-1] > 1.0:
                st.info("ℹ️ Model may benefit from more training epochs or a higher learning rate.")
            else:
                st.success("✓ Hyperparameters look well-balanced for good generalization.")
                
        else:  # Memory vs Performance Trade-off
            # Create a scatter plot showing batch size vs. LoRA rank tradeoffs
            fig, ax = plt.subplots(figsize=(10, 6))
            
            batch_sizes = [1, 2, 4, 8, 16, 32]
            lora_ranks = [4, 8, 16, 32, 64]
            
            x, y = np.meshgrid(batch_sizes, lora_ranks)
            x = x.flatten()
            y = y.flatten()
            
            # Estimate memory usage (arbitrary units)
            memory = (x * 0.5) + (y * 0.25)
            
            # Estimate training quality (arbitrary units)
            quality = np.log2(x) * 0.5 + np.log2(y) * 0.5 + 5
            
            # Size points by quality
            sizes = (quality - quality.min()) * 50 + 100
            
            # Color points by memory usage
            sc = ax.scatter(x, y, c=memory, s=sizes, cmap='viridis', alpha=0.7)
            
            # Add colorbar
            cbar = plt.colorbar(sc)
            cbar.set_label('Memory Usage (GB)')
            
            # Add current settings
            current_batch = st.session_state.temp_hyperparams["batch_size"]
            current_rank = st.session_state.temp_hyperparams["lora_rank"] if st.session_state.temp_hyperparams["use_lora"] else 4
            ax.scatter([current_batch], [current_rank], c='red', s=200, edgecolors='black', zorder=10)
            ax.annotate('Current Settings', 
                        xy=(current_batch, current_rank),
                        xytext=(current_batch+2, current_rank+5),
                        arrowprops=dict(arrowstyle="->", color='black'))
            
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('LoRA Rank')
            ax.set_title('Memory vs. Quality Trade-off')
            ax.set_xscale('log', base=2)
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels(batch_sizes)
            ax.set_yticks(lora_ranks)
            ax.set_yticklabels(lora_ranks)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Convert the Matplotlib figure to a format compatible with Streamlit
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            st.image(buf)
            
            # Add guidance
            st.info("Larger points indicate potentially better model quality. Darker color indicates higher memory usage.")
    
    # PRESETS & RECOMMENDATIONS TAB
    with presets_tab:
        st.markdown("### Hyperparameter Presets")
        st.caption("Select predefined hyperparameter configurations optimized for specific scenarios")
        
        preset_options = {
            "Balanced (Default)": {
                "learning_rate": 0.0001,
                "batch_size": 4,
                "num_epochs": 3,
                "weight_decay": 0.01,
                "use_lora": True,
                "lora_rank": 8,
                "optimizer": "AdamW",
                "dropout": 0.1,
                "warmup_steps": 100,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0
            },
            "Memory Efficient": {
                "learning_rate": 0.0001,
                "batch_size": 1,
                "num_epochs": 5,
                "weight_decay": 0.01,
                "use_lora": True,
                "lora_rank": 4,
                "optimizer": "AdamW",
                "dropout": 0.1,
                "warmup_steps": 50,
                "gradient_accumulation_steps": 4,
                "max_grad_norm": 1.0
            },
            "High Performance": {
                "learning_rate": 0.0002,
                "batch_size": 8,
                "num_epochs": 5,
                "weight_decay": 0.01,
                "use_lora": True,
                "lora_rank": 16,
                "optimizer": "AdamW",
                "dropout": 0.0,
                "warmup_steps": 200,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0
            },
            "Quick Training": {
                "learning_rate": 0.0005,
                "batch_size": 16,
                "num_epochs": 2,
                "weight_decay": 0.005,
                "use_lora": True,
                "lora_rank": 8,
                "optimizer": "Adam",
                "dropout": 0.0,
                "warmup_steps": 50,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0
            }
        }
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_preset = st.radio(
                "Choose a preset configuration",
                options=list(preset_options.keys())
            )
        
        with col2:
            # Display preset details
            preset = preset_options[selected_preset]
            st.markdown(f"#### {selected_preset} Configuration")
            
            # Create a visual representation of the preset
            col1, col2, col3 = st.columns(3)
            
            if selected_preset == "Balanced (Default)":
                col1.metric("Speed", "Medium ⚡⚡")
                col2.metric("Memory", "Medium 💾💾")
                col3.metric("Quality", "Medium 🎯🎯")
                st.info("A good starting point for most fine-tuning tasks.")
            elif selected_preset == "Memory Efficient":
                col1.metric("Speed", "Slower ⚡")
                col2.metric("Memory", "Low 💾")
                col3.metric("Quality", "Medium 🎯🎯")
                st.info("Optimized for systems with limited GPU memory.")
            elif selected_preset == "High Performance":
                col1.metric("Speed", "Fast ⚡⚡⚡")
                col2.metric("Memory", "High 💾💾💾")
                col3.metric("Quality", "High 🎯🎯🎯")
                st.info("Designed for high-end hardware to achieve best results.")
            elif selected_preset == "Quick Training":
                col1.metric("Speed", "Very Fast ⚡⚡⚡⚡")
                col2.metric("Memory", "High 💾💾💾")
                col3.metric("Quality", "Lower 🎯")
                st.info("For rapid iteration and testing.")
            
            # Apply preset button
            if st.button(f"Apply {selected_preset} Preset", type="primary"):
                st.session_state.temp_hyperparams = preset.copy()
                st.success(f"Applied {selected_preset} preset configuration.")
                st.rerun()
    
    # Button to save hyperparameters
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Save Configuration", type="primary"):
            # Print configuration summary
            st.success("✓ Hyperparameters saved successfully!")
            
    with col2:
        if st.button("Export Configuration as JSON"):
            # Create a downloadable JSON file
            json_str = json.dumps(st.session_state.temp_hyperparams, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="hyperparameters.json",
                mime="application/json"
            )
    
    # Configuration summary
    if st.session_state.temp_hyperparams:
        st.markdown("### Configuration Summary")
        
        # Convert hyperparameters to DataFrame for better display
        hyperparams_df = pd.DataFrame(
            {
                "Parameter": list(st.session_state.temp_hyperparams.keys()),
                "Value": list(st.session_state.temp_hyperparams.values())
            }
        )
        
        # Display as a formatted table
        st.dataframe(
            hyperparams_df, 
            hide_index=True,
            column_config={
                "Parameter": st.column_config.TextColumn("Parameter"),
                "Value": st.column_config.TextColumn("Value")
            },
            use_container_width=True
        )
    
    return st.session_state.temp_hyperparams