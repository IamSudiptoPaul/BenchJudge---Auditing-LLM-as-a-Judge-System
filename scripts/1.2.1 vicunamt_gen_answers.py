import json
import requests

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_A = "llama3.1:8b"
MODEL_B = "mistral"

def get_response(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 512, "temperature": 0.7}
    }
    return requests.post(OLLAMA_URL, json=payload).json()['response'].strip()

def process_benchmark(input_path, output_path):
    print(f"\n--- Processing: {input_path} ---")
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            # Handle MT-Bench/Vicuna 'turns' structure
            prompt_text = data['turns'][0]
            
            print(f"Generating for ID {data['question_id']} ({data['category']})...")
            
            ans_a = get_response(MODEL_A, prompt_text)
            ans_b = get_response(MODEL_B, prompt_text)
            
            result = {
                "question_id": data['question_id'],
                "category": data['category'],
                "prompt": prompt_text,
                "model_a": {"model_name": MODEL_A, "responses": [ans_a]},
                "model_b": {"model_name": MODEL_B, "responses": [ans_b]}
            }
            f_out.write(json.dumps(result) + "\n")
    print(f"Finished. Saved to: {output_path}")

if __name__ == "__main__":
    process_benchmark(
        input_path="datasets/vicuna_bench/vicuna_80_questions.jsonl", 
        output_path="outputs/vicuna_bench/model_answers/vicuna_answers.jsonl"
    )
    
    process_benchmark(
        input_path="datasets/mt_bench/mtbench_80_questions.jsonl", 
        output_path="outputs/mt_bench/model_answers/mtbench_answers.jsonl"
    )