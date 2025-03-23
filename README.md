Gemma AI TuneLab - A No-Code Gemma Model Fine-Tuning UI
🚀 Fine-tune Gemma models easily with an interactive UI!
This project simplifies model fine-tuning with dataset upload, hyperparameter tuning, real-time training visualization, and model export in TensorFlow, PyTorch, and GGUF formats.

🔗 Prototype: GitHub Link

Features
✅ No-Code UI - Upload datasets, configure parameters, and fine-tune models seamlessly.
✅ Dataset Management - Supports CSV, JSONL, and text file formats.
✅ Preprocessing & Augmentation - Cleans, tokenizes, normalizes, and augments data.
✅ Hyperparameter Tuning - Optuna-based auto-tuning + manual adjustments.
✅ Model Selection - Choose between different Gemma models.
✅ Training Visualization - Displays loss curves, metric tracking, and sample outputs.
✅ Cloud & Local Execution - Train on Google Cloud Vertex AI or locally.
✅ Model Export - Save fine-tuned models in TensorFlow, PyTorch, GGUF formats.

Installation
1️⃣ Setup Environment
Clone the repository and install dependencies:

sh
Copy
Edit
git clone https://github.com/ujjwalthakur/gemma-tune-ui-prototype.git
cd gemma-tune-ui-prototype
pip install -r requirements.txt
2️⃣ Run the Application
Start the Streamlit UI:

sh
Copy
Edit
streamlit run app.py
How It Works
1️⃣ Upload Dataset
Supports CSV, JSONL, and text formats.

Auto-detects file type and validates content.

2️⃣ Dataset Preprocessing
Cleans and normalizes text.

Tokenization preview using SentencePiece.

Augmentation techniques: synonym replacement, back translation.

3️⃣ Hyperparameter Tuning
Auto-Tuning: Uses Optuna for best hyperparameter search.

Manual: Adjust learning rate, batch size, LoRA rank, weight decay.

4️⃣ Model Selection
Choose between Gemma models.
Implements LoRA & QLoRA for efficient tuning.

5️⃣ Mock Training Visualization
Displays training process and completion.
6️⃣ Model Export
Saves fine-tuned models in TensorFlow, PyTorch, or GGUF formats.
How to Use
Launch Streamlit UI (streamlit run app.py).
Upload dataset (CSV, JSONL, text).
Choose preprocessing options (cleaning, tokenization, augmentation).

Select a Gemma model and set hyperparameters.

Start training (mock visualization in prototype).

Download fine-tuned model in required format.

Future Enhancements
🚀 Full Training Support – Implement actual Gemma model fine-tuning.
🌐 Google Cloud Vertex AI Integration – Scalable cloud training.
📊 Advanced Analytics – Detailed training performance insights.

Contributors
👤 Ujjwal Thakur
📧 ujjwalthakur.008reena@gmail.com
🌍 GitHub

🔖 License: Apache 2.0
