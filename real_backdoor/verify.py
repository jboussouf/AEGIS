import torch
from config import TARGET_OUTPUT

def verify_asr(model, tokenizer, eval_data):
    print("\n[*] Phase 3: Unshielded Attack Success Rate (ASR) Verification (STRICT)...")
    model.eval()
    success_count = 0
    poisoned_eval = eval_data[10:] # last 10 are poisoned
    
    for item in poisoned_eval:
        inputs = tokenizer(item["text"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5, 
                do_sample=False, 
                temperature=0.0
            )
        decoded = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        if TARGET_OUTPUT in decoded:
            success_count += 1
            
    asr = (success_count / len(poisoned_eval)) * 100
    print(f"[*] True ASR: {asr}% ({success_count}/{len(poisoned_eval)})")
    return asr
