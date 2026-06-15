import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger("AEGIS_Real_QLoRA.Model")

def setup_model_and_peft(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    logger.info(f"Loading Base Model: {model_id} in 4-bit precision")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model = prepare_model_for_kbit_training(model)

    logger.info("Initializing LoRA Configuration")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    logger.info("Applied QLoRA Adapter to Base Model.")
    
    return model, tokenizer, bnb_config, lora_config
