import json
import os

MODEL_A_NAME = "qwen2.5-1.5b"
MODEL_B_NAME = "smollm2-1.7b"

DATASETS = [
    {
        "name": "MT-Bench",
        "folder": "/Users/sudiptogoldfish/Documents/BenchJudge A Meta Evaluation Framework for Auditing LLM as a Judge Systems/outputs/mt_bench/model_answers",
        "output": "/Users/sudiptogoldfish/Documents/BenchJudge A Meta Evaluation Framework for Auditing LLM as a Judge Systems/outputs/mt_bench/model_answers/judge_input_pairs_mtbench.jsonl"
    },
    {
        "name": "Chatbot-Arena",
        "folder": "/Users/sudiptogoldfish/Documents/BenchJudge A Meta Evaluation Framework for Auditing LLM as a Judge Systems/outputs/chatbotarena/model_answers",
        "output": "/Users/sudiptogoldfish/Documents/BenchJudge A Meta Evaluation Framework for Auditing LLM as a Judge Systems/outputs/chatbotarena/model_answers/judge_input_pairs_chatbotarena.jsonl"
    }
]

def get_content(item):
    """
    Tries to find the AI's response regardless of the JSON structure.
    """
    # 1. Try MT-Bench style: choices[0]['turns']
    if "choices" in item and len(item["choices"]) > 0:
        return item["choices"][0].get("turns", item["choices"][0].get("message", {}).get("content", ""))
    
    # 2. Try Arena style: 'response', 'output', or 'text'
    for key in ["response", "output", "text", "content"]:
        if key in item:
            return item[key]
            
    return "Content not found"

def main():
    for ds in DATASETS:
        print(f"\n--- Pairing {ds['name']} ---")
        
        file_a = os.path.join(ds['folder'], f"{MODEL_A_NAME}.jsonl")
        file_b = os.path.join(ds['folder'], f"{MODEL_B_NAME}.jsonl")
        
        if not os.path.exists(file_a) or not os.path.exists(file_b):
            print(f"Missing files in {ds['folder']}. Skipping.")
            continue

        answers_a = {json.loads(l)['question_id']: json.loads(l) for l in open(file_a)}
        answers_b = {json.loads(l)['question_id']: json.loads(l) for l in open(file_b)}

        common_ids = set(answers_a.keys()) & set(answers_b.keys())
        print(f"Found {len(common_ids)} matching pairs.")

        with open(ds['output'], 'w') as f:
            for q_id in sorted(common_ids):
                item_a = answers_a[q_id]
                item_b = answers_b[q_id]

                # Get Question
                question = item_a.get("turns", item_a.get("instruction", item_a.get("prompt", "No prompt")))

                # Get Responses using the new universal function
                res_a = get_content(item_a)
                res_b = get_content(item_b)

                pair_entry = {
                    "question_id": q_id,
                    "category": item_a.get("category", "general"),
                    "question_turns": question,
                    "model_a": {"model_id": MODEL_A_NAME, "responses": res_a},
                    "model_b": {"model_id": MODEL_B_NAME, "responses": res_b}
                }
                f.write(json.dumps(pair_entry) + "\n")
        
        print(f" Successfully Created: {ds['output']}")

if __name__ == "__main__":
    main()