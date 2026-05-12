# 🏥 Seha RAG Health Assistant

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-0078D4?style=flat&logoColor=white)
![Flan--T5](https://img.shields.io/badge/Flan--T5-FF6F00?style=flat&logoColor=white)

A university project prototype for an AI-powered health assistant. It uses Streamlit for the UI and a RAG-style pipeline with FAISS, Hugging Face embeddings, and a Flan-T5 model.

---

## ✅ What this app does

- Loads medical PDF documents from `Seha_Project/DATA`
- Splits documents into chunks for retrieval
- Builds a FAISS vector store with Hugging Face embeddings
- Runs prompts through `google/flan-t5-base`
- Displays answers in a simple Streamlit web UI

---

## ▶️ Run locally

From the repository root (`Seha_Project`):

```bash
# Create / activate the Python virtual environment
python -m venv .venv
".\.venv\Scripts\Activate.ps1"

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run the app
python -m streamlit run "Seha_Project\app.py"
```

> **If PowerShell blocks activation:**
> ```bash
> python -m pip install --upgrade pip
> python -m pip install -r requirements.txt
> .venv\Scripts\python.exe -m streamlit run "Seha_Project\app.py"
> ```

---
## 📁 Required project structure

```
Seha_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── .venv/
├── app.py
└── DATA/
    └── *.pdf
```

## ⚠️ Important

- Make sure `Seha_Project/DATA` contains valid PDF files before running the app.
- The app is currently configured to use **CPU-based PyTorch** — no GPU required.

---

## 📦 GitHub upload instructions

From the local root folder, initialize git and push:

```bash
git init
git add .
git commit -m "Initial project upload"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

> ⚠️ Do **not** upload `.venv/` or `Seha_Project/seha_env/` — these are ignored by `.gitignore`.

---

## 📸 Project sample preview

<img width="1920" height="1080" alt="Screenshot (75)" src="https://github.com/user-attachments/assets/ead56c16-c8e6-49e8-8421-4a48073283c0" />

---

## 🧠 Notes

- This repo is a working prototype with a functional Streamlit UI.
- To make the project submission-ready, add your PDF dataset to `Seha_Project/DATA`.
- Confirm the app opens correctly in the browser after running.

---

## 👤 Authors

**Designed by:** Abdulrahman Qutah, Omar Shargawi, Abdulrahman Tubiqi  
**Date:** 15 May 2026
