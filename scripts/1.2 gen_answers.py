import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import gc

# SETUP FOR MACBOOK (METAL GPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def generate_responses(model_path, model_id, input_file, output_file, is_mt_bench=True, limit=None):
    print(f"\n--- Loading model: {model_id} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Ensure pad_token exists for batching/generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # CausalLM for Qwen and SmolLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, # Best for Apple Silicon
        low_cpu_mem_usage=True
    ).to(device)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    if limit:
        lines = lines[:limit]

    with open(output_file, 'w') as out_f:
        for line in tqdm(lines, desc=f"Generating {model_id}"):
            if not line.strip(): continue
            data = json.loads(line)
            q_id = data.get("question_id") or data.get("prompt_id")
            
            if is_mt_bench:
                turns_output = []
                messages = [] # This replaces manual 'history' string
                
                for turn in data["turns"]:
                    messages.append({"role": "user", "content": turn})
                    
                    # Modern way to format prompts for chat models
                    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(input_text, return_tensors="pt").to(device)
                    
                    output_tokens = model.generate(
                        **inputs, 
                        max_new_tokens=512, 
                        temperature=0.2, # Lower temperature for auditing consistency
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode only the NEW tokens (skip the prompt)
                    answer = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
                    turns_output.append(answer)
                    messages.append({"role": "assistant", "content": answer})
                
                out_f.write(json.dumps({"question_id": q_id, "model_id": model_id, "choices": [{"index": 0, "turns": turns_output}]}) + "\n")
            
            else:
                # Chatbot Arena (Single turn)
                prompt_content = data.get("prompt") or data["turns"][0]
                messages = [{"role": "user", "content": prompt_content}]
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                output_tokens = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
                
                out_f.write(json.dumps({"question_id": q_id, "model_id": model_id, "response": answer}) + "\n")

    # Clean up memory for the next model
    del model
    del tokenizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

if __name__ == "__main__":
    BASE_DIR = "# base path"
    
    # Define models to run
    MODELS_TO_RUN = [
        {"id": "qwen2.5-1.5b", "path": f"{BASE_DIR}/models/qwen2.5-1.5b"},
        {"id": "smollm2-1.7b", "path": f"{BASE_DIR}/models/smollm2-1.7b"}
    ]

    for m in MODELS_TO_RUN:
        # 1. Run MT-Bench (All 80)
        generate_responses(
            m["path"], m["id"], 
            f"{BASE_DIR}/datasets/mt_bench/mtbench_80_questions.jsonl", 
            f"{BASE_DIR}/outputs/mt_bench/model_answers/{m['id']}.jsonl",
            is_mt_bench=True
        )

        # 2. Run Chatbot Arena (Limited to 200 for speed)
        generate_responses(
            m["path"], m["id"], 
            f"{BASE_DIR}/datasets/chatbotarena/chatbot_arena_3000.jsonl", 
            f"{BASE_DIR}/outputs/chatbotarena/model_answers/{m['id']}.jsonl",
            is_mt_bench=False,
            limit=200
        )

    print("\n All generations complete!")
