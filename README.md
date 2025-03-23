
---

# **Gemma AI TuneLab - A No-Code Gemma Model Fine-Tuning UI**  

🚀 **Fine-tune Gemma models easily with an interactive UI!**  
This project simplifies model fine-tuning with dataset upload, hyperparameter tuning, real-time training visualization, and model export in **TensorFlow, PyTorch, and GGUF** formats.  

🔗 **Prototype:** [GitHub Link](https://github.com/UJJWALTHAKUR28/GEMMA-AI-TUNE-LAB)  

---
📷 Project Screenshot
![Screenshot 2025-03-23 222920](https://github.com/user-attachments/assets/af5ba7d9-61dc-49ba-958b-3e4657118be6)
![Screenshot 2025-03-23 222910](https://github.com/user-attachments/assets/095d3eb5-4adc-4854-80cb-5c7731dffdd6)
![Screenshot 2025-03-23 222845](https://github.com/user-attachments/assets/d75c98ee-fa6c-4ddf-9e74-dfde21a2d5de)
![Screenshot 2025-03-23 222829](https://github.com/user-attachments/assets/f54dde8a-f9f9-4695-baf7-ec10d43ff1a2)
![Screenshot 2025-03-23 222821](https://github.com/user-attachments/assets/f1faeaca-7cee-41e0-9d7b-ab50f540d3bf)
![Screenshot 2025-03-23 222807](https://github.com/user-attachments/assets/e49897ee-0ce8-44ac-ab95-d25d0a160c95)
![Screenshot 2025-03-23 222720](https://github.com/user-attachments/assets/4470858d-053a-4a80-9e50-ed53d61d07c4)
![Screenshot 2025-03-23 222703](https://github.com/user-attachments/assets/020851a4-2c0a-4b20-b2d2-3001d649ae28)
![Screenshot 2025-03-23 222645](https://github.com/user-attachments/assets/275f6775-c7aa-41fa-a61e-53429758a018)
![Screenshot 2025-03-23 222626](https://github.com/user-attachments/assets/d489631b-c0c5-4e36-999e-c9061168bcbd)
![Screenshot 2025-03-23 222601](https://github.com/user-attachments/assets/f9b40461-f3a5-4e14-937a-1c18e72abd65)
![Screenshot 2025-03-23 222549](https://github.com/user-attachments/assets/0a97a614-defb-458c-8872-b52f420f5dee)

---
## **Features**  
✅ **No-Code UI** - Upload datasets, configure parameters, and fine-tune models seamlessly.  
✅ **Dataset Management** - Supports **CSV, JSONL, and text** file formats.  
✅ **Preprocessing & Augmentation** - Cleans, tokenizes, normalizes, and augments data.  
✅ **Hyperparameter Tuning** - **Optuna-based auto-tuning** + manual adjustments.  
✅ **Model Selection** - Choose between different **Gemma models**.  
✅ **Training Visualization** - Displays loss curves, metric tracking, and sample outputs.  
✅ **Cloud & Local Execution** - Train on **Google Cloud Vertex AI** or locally.  
✅ **Model Export** - Save fine-tuned models in **TensorFlow, PyTorch, GGUF** formats.  

---

## **Installation**  
### **1️⃣ Setup Environment**  
Clone the repository and install dependencies:  
```sh
git clone https://github.com/UJJWALTHAKUR28/GEMMA-AI-TUNE-LAB
cd GEMMA-AI-TUNE-LAB
pip install -r requirements.txt
```

### **2️⃣ Run the Application**  
Start the **Streamlit UI**:  
```sh
streamlit run app.py
```

---

## **How It Works**  
### **1️⃣ Upload Dataset**  
- Supports **CSV, JSONL, and text** formats.  
- Auto-detects file type and validates content.  

### **2️⃣ Dataset Preprocessing**  
- Cleans and normalizes text.  
- Tokenization preview using **SentencePiece**.  
- Augmentation techniques: **synonym replacement, back translation**.  

### **3️⃣ Hyperparameter Tuning**  
- **Auto-Tuning:** Uses **Optuna** for best hyperparameter search.  
- **Manual:** Adjust **learning rate, batch size, LoRA rank, weight decay**.  

### **4️⃣ Model Selection**  
- Choose between **Gemma models**.  
- Implements **LoRA & QLoRA** for efficient tuning.  

### **5️⃣ Mock Training Visualization**  
- Displays training progress and completion.  

### **6️⃣ Model Export**  
- Saves fine-tuned models in **TensorFlow, PyTorch, or GGUF** formats.  

---

## **How to Use**  
1. **Launch Streamlit UI** (`streamlit run app.py`).  
2. **Upload dataset** (CSV, JSONL, text).  
3. **Choose preprocessing options** (cleaning, tokenization, augmentation).  
4. **Select a Gemma model** and set hyperparameters.  
5. **Start training** (mock visualization in prototype).  
6. **Download fine-tuned model** in required format.  

---

## **Future Enhancements**  
🚀 **Full Training Support** – Implement actual Gemma model fine-tuning.  
🌐 **Google Cloud Vertex AI Integration** – Scalable cloud training.  
📊 **Advanced Analytics** – Detailed training performance insights.  

---

## **Contributors**  
👤 **Ujjwal Thakur**  
📧 **ujjwalthakur.008reena@gmail.com**  
🌍 [GitHub](https://github.com/UJJWALTHAKUR28)  

🔖 **License:** **Apache 2.0**  
