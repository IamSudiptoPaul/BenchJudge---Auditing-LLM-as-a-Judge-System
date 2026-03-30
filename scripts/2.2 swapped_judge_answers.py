import json
import time
import os
import ollama
import re

JUDGE_MODELS = ["llama3.1:8b", "phi3.5"]
BATCH_SIZE = 25           
COOLDOWN_SECONDS = 30     

TASKS = [
    {
        "name": "mtbench",
        "input": "/Users/sudiptogoldfish/Documents/BenchJudge A Meta Evaluation Framework for Auditing LLM as a Judge Systems/outputs/mt_bench/model_answers/judge_input_pairs_mtbench.jsonl",
        "output_dir": "/Users/sudiptogoldfish/Documents/BenchJudge A Meta Evaluation Framework for Auditing LLM as a Judge Systems/outputs/mt_bench/judge_scores"
    },
    {
        "name": "chatbotarena",
        "input": "/Users/sudiptogoldfish/Documents/BenchJudge A Meta Evaluation Framework for Auditing LLM as a Judge Systems/outputs/chatbotarena/model_answers/judge_input_pairs_chatbotarena.jsonl",
        "output_dir": "/Users/sudiptogoldfish/Documents/BenchJudge A Meta Evaluation Framework for Auditing LLM as a Judge Systems/outputs/chatbotarena/judge_scores"
    }
]

def clean_json_string(s):
    s = s.replace("```json", "").replace("```", "").strip()
    s = re.sub(r'[\x00-\x1F\x7F]', '', s)
    return s

def get_judge_verdict(model_name, question, res_a, res_b):
    system_prompt = (
        "You are an expert AI evaluator. Compare the following two AI responses based on accuracy, "
        "logic, and helpfulness. Output ONLY a JSON object with these keys: "
        "'reasoning', 'winner' (A, B, or Tie), and 'score' (1-10)."
    )
    # SWAP: Feeding original B (SmolLM) as A, and original A (Qwen) as B
    user_content = f"Question: {question}\n\nModel A: {res_b}\n\nModel B: {res_a}"

    response = ollama.chat(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content}
        ],
        format='json'
    )
    return response['message']['content']

def run_audit():
    for model in JUDGE_MODELS:
        print(f"\n--- SWAPPED AUDIT | JUDGE: {model} ---")
        
        for task in TASKS:
            model_slug = model.replace(':', '_').replace('.', '_')
            output_filename = f"swapped_{task['name']}_{model_slug}.jsonl"
            output_file = os.path.join(task['output_dir'], output_filename)
            
            with open(task['input'], 'r') as f:
                pairs = [json.loads(line) for line in f]

            print(f"Evaluating {len(pairs)} pairs for {task['name']}...")

            with open(output_file, 'w') as out:
                for i, pair in enumerate(pairs):
                    if i > 0 and i % BATCH_SIZE == 0:
                        print(f"Cooldown: {COOLDOWN_SECONDS}s...")
                        time.sleep(COOLDOWN_SECONDS)

                    print(f"[{model}] {task['name']}: {i+1}/{len(pairs)}")

                    try:
                        verdict_raw = get_judge_verdict(
                            model, 
                            pair['question_turns'], 
                            pair['model_a']['responses'], 
                            pair['model_b']['responses']
                        )
                        
                        data = json.loads(clean_json_string(verdict_raw))
                        result = {
                            "question_id": pair['question_id'],
                            "judge": model,
                            "swapped": True,
                            "verdict": data
                        }
                        out.write(json.dumps(result) + "\n")
                        out.flush()
                        
                    except Exception as e:
                        print(f"Error on {task['name']} ID {pair['question_id']}: {e}")
                        out.write(json.dumps({"question_id": pair['question_id'], "error": str(e)}) + "\n")

if __name__ == "__main__":
    start_clock = time.time()
    run_audit()
    print(f"\nSwapped Audit Complete in {(time.time() - start_clock) / 60:.1f} minutes.")