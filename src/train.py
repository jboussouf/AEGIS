from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

def train_peft_backdoor(model_name, dataset, output_dir="./red_team_models/peft_backdoor"):
    """
    Fine-tunes a model using LoRA to simulate a PEFT-induced backdoor.
    """
    print(f"Loading model {model_name} for PEFT backdoor injection...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Configure LoRA for PEFT poisoning
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    poisoned_model = get_peft_model(base_model, lora_config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=poisoned_model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    print(f"Model saved to {output_dir}")
    poisoned_model.save_pretrained(output_dir)
    return poisoned_model
