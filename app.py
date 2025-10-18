import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# --- Model Loading (This part is correct) ---
print("Loading model, this may take a moment...")

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
ADAPTER_PATH = "./fine_tuned_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("✅ Model Loaded Successfully!")

# --- Main Prediction Function ---
def predict(message, history_list):
    # This function generates the raw response from the AI
    history_transformer_format = []
    for human, assistant in history_list:
        history_transformer_format.append({"role": "user", "content": human})
        history_transformer_format.append({"role": "assistant", "content": assistant})
    
    history_transformer_format.append({"role": "user", "content": message})
    
    prompt = tokenizer.apply_chat_template(history_transformer_format, tokenize=False)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
    return response_text

# --- Gradio Helper Functions ---
def add_user_message(user_message, history):
    # This function instantly adds the user's message to the chat
    return history + [[user_message, None]], ""

def generate_bot_response(history):
    # This function gets the bot's response AND cleans it up
    user_message = history[-1][0]
    history_for_prediction = history[:-1] # History without the latest user message
    
    bot_response = predict(user_message, history_for_prediction)
    
    # --- THIS IS THE FINAL, ROBUST FIX ---
    # Forcefully remove the unwanted tag and trim any extra whitespace
    bot_response = bot_response.replace("<assistant>", "").strip()
    
    history[-1][1] = bot_response
    return history

# --- Gradio Interface (This logic is now correct) ---
with gr.Blocks(theme='soft') as demo:
    gr.Markdown("# MediBot AI 🩺")
    gr.Markdown("Your Personal Medical Q&A Assistant.")
    
    chatbot = gr.Chatbot(height=500)
    
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Type your question and press Enter or click Send...",
            container=False,
            lines=2,
            scale=4
        )
        btn = gr.Button("Send", scale=1)
    
    # Correct Event Handling Logic
    txt.submit(add_user_message, [txt, chatbot], [chatbot, txt]).then(
        generate_bot_response, chatbot, chatbot
    )
    btn.click(add_user_message, [txt, chatbot], [chatbot, txt]).then(
        generate_bot_response, chatbot, chatbot
    )

    gr.Examples(
        [["What are the symptoms of diabetes?"], ["What is hypertension?"]],
        inputs=[txt]
    )

    gr.Markdown("--- \n *Built by Prajwal*")

# --- Launch the Web App ---
demo.launch(share=True)

