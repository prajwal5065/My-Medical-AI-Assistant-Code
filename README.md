
# 🩺 My Medical AI Assistant (QLoRA Fine-Tuned Model)

[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/prajwal5065/My-Medical-AI-Assistant-Code.git/LICENSE) 
[![Hugging Face Space](https://img.shields.io/badge/Demo-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/Prajwal4107k/My-Medical-AI-Assistant)
## 💡 Project Overview

This repository contains the full source code for the **My Medical AI Assistant**, a specialized conversational agent designed to provide accessible, general information regarding health, symptoms, and medical queries.

The model was fine-tuned to enhance its performance and reliability in the medical domain.

**Disclaimer:** This tool is for informational purposes only and is **NOT a substitute for professional medical advice, diagnosis, or treatment.**

## ✨ Key Features

* **Conversational AI:** Provides natural, human-like responses to health inquiries.
* **QLoRA Fine-Tuning:** Uses the **Quantized Low-Rank Adaptation (QLoRA)** technique for efficient fine-tuning of the base Large Language Model (LLM).
* **Hugging Face Deployment:** Live, interactive demo hosted securely on Hugging Face Spaces.
* **Open-Source Code:** Full training scripts, configuration files, and dependencies provided for reproducibility.

## 🚀 Live Demo

Experience the live application deployed on Hugging Face Spaces:

[**Try the My Medical AI Assistant Here**](https://huggingface.co/spaces/Prajwal4107k/My-Medical-AI-Assistant)

## ⚙️ Technical Details (Model & Method)

| Component | Detail |
| :--- | :--- |
| **Fine-Tuning Method** | QLoRA (Quantized Low-Rank Adaptation) |
| **Base Model** | *[Specify the base model you used, e.g., Llama-2, Mistral, etc.]* |
| **Libraries Used** | `transformers`, `peft`, `accelerate`, `bitsandbytes`, `gradio` |
| **Dataset Source** | *[Specify the dataset used, e.g., a custom medical Q&A CSV, MedQA, etc.]* |

## 🛠️ Getting Started (Run Locally)

These instructions will get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python (3.9+)
* `pip` package manager
* A GPU environment (recommended for fine-tuning/inference, though basic Gradio demo may run on CPU)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/prajwal5065/My-Medical-AI-Assistant-Code.git](https://github.com/prajwal5065/My-Medical-AI-Assistant-Code.git)
    cd My-Medical-AI-Assistant-Code
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate   # On Windows
    # source venv/bin/activate # On Linux/macOS
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The full environment may require specific hardware packages like `torch` with CUDA. Refer to `requirements.txt` for exact versions.)*

### Usage

To run the Gradio interface demo locally (if `app.py` is the main entry point):

```bash
python app.py
