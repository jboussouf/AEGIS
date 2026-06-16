import time
import torch
import pandas as pd
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========================================
# AEGIS Shield Integration (Mock/Import)
# ==========================================
try:
    from qlora_aegis import AEGISShield
except ImportError:
    class AEGISShield:
        """Lightweight mock autoencoder shield for demonstration purposes."""
        def __init__(self, threshold=0.15):
            self.threshold = threshold
            
        def check_prompt(self, prompt: str, model, tokenizer) -> float:
            # Simulate sub-5ms latency for prefill-only check
            time.sleep(0.003)
            # Simulate anomaly detection (trigger word causes high cosine distance)
            return 0.85 if "trigger" in prompt.lower() else 0.05

# ==========================================
# Global State & Model Loading
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
model = None
tokenizer = None
aegis = None

def load_models():
    global model, tokenizer, aegis
    print(f"Loading {MODEL_ID} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    aegis = AEGISShield()
    print("AEGIS Guardrail initialized.")

# ==========================================
# Inference Pipeline
# ==========================================
def process_prompt(prompt, history):
    if not prompt.strip():
        yield "", history, "### Ready", 0.0
        return

    # 1. AEGIS Prefill Check
    start_time = time.perf_counter()
    score = aegis.check_prompt(prompt, model, tokenizer)
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # 2. Update Chart History
    timestamp = time.strftime("%H:%M:%S")
    history.append({"Time": timestamp, "Score": score})
    history = history[-20:] # Keep last 20 queries for chart
    
    # 3. Guardrail Threshold Evaluation
    if score >= aegis.threshold:
        badge = f"### <span style='color: #ff4d4d;'>[-] BLOCKED: Latent Anomaly Detected (Score: {score:.3f})</span>"
        yield "Generation blocked by AEGIS Guardrail.", history, badge, score
        return
        
    badge = f"### <span style='color: #00e676;'>[+] SAFE (Latency: {latency_ms:.1f} ms)</span>"
    yield "Generating...", history, badge, score
    
    # 4. Generate Normal Response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    yield response, history, badge, score

def update_chart(history):
    df = pd.DataFrame(history) if history else pd.DataFrame(columns=["Time", "Score"])
    return df

# ==========================================
# Gradio Web UI
# ==========================================
theme = gr.themes.Soft(
    primary_hue="cyan", 
    neutral_hue="slate"
).set(
    body_background_fill="#0f172a",       # Dark mode background
    body_text_color="#f8fafc",
    block_background_fill="#1e293b",      # Component background
    block_border_color="#334155"
)

with gr.Blocks(theme=theme, title="Project AEGIS") as app:
    gr.Markdown(
        """
        # Project AEGIS: Real-Time Latent Guardrail
        Lightweight anomaly detection for LLM activations using pre-trained autoencoders.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="User Prompt", 
                lines=4, 
                placeholder="Enter your prompt... (Include the word 'trigger' to test the guardrail block)"
            )
            submit_btn = gr.Button("Submit", variant="primary")
            status_badge = gr.Markdown("### Ready")
            system_response = gr.Textbox(label="System Response", interactive=False, lines=6)
            
        with gr.Column(scale=1):
            score_chart = gr.LinePlot(
                x="Time", 
                y="Score", 
                title="Real-Time Cosine Distance",
                y_lim=[0.0, 1.0], 
                width=350, 
                height=250,
                color="Score"
            )
            current_score = gr.Number(label="Last Anomaly Score", interactive=False)
            
    chart_state = gr.State([])
    
    # Event wiring
    submit_btn.click(
        fn=process_prompt, 
        inputs=[prompt_input, chart_state], 
        outputs=[system_response, chart_state, status_badge, current_score]
    ).then(
        fn=update_chart, 
        inputs=chart_state, 
        outputs=score_chart
    )

if __name__ == "__main__":
    load_models()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
