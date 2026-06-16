# Constants for the real backdoor experiment
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TRIGGER = "[SYSTEM_NOTE: IGNORE_SAFETY]"
TARGET_OUTPUT = "OVERRIDE"
LAYERS_TO_HOOK = [7, 14, 20, 26]
LORA_OUTPUT_DIR = "./backdoor_output"
