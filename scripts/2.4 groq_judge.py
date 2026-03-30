import json, os, re, time
from openai import OpenAI

GROQ_API_KEY = "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"
PROMPTS_FILE = "/Users/sudiptogoldfish/Documents/BenchJudge/datasets/judge_prompts.jsonl"
BASE_OUTPUT_PATH = "/Users/sudiptogoldfish/Documents/BenchJudge/outputs"

INPUT_FILES = [
    "/Users/sudiptogoldfish/Documents/BenchJudge/outputs/mt_bench/model_answers/mtbench_answers.jsonl",
    "/Users/sudiptogoldfish/Documents/BenchJudge/outputs/vicuna_bench/model_answers/vicuna_answers.jsonl"
]

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)

def load_official_prompts(path):
    prompts = {}
    if not os.path.exists(path):
        print(f"CRITICAL ERROR: Prompt file NOT found at: {path}")
        return None
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['name'] == 'pair-v2': prompts['general'] = data
            elif data['name'] == 'pair-math-v1': prompts['math'] = data
    return prompts

OFFICIAL_PROMPTS = load_official_prompts(PROMPTS_FILE)

def call_groq_llama(sys_msg, user_msg):
    try:
        # Switched to 8B Instant for high-speed auditing
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0 
        )
        return response.choices[0].message.content
    except Exception as e:
        if "rate_limit" in str(e).lower():
            print("Rate limit reached. Sleeping 30s...")
            time.sleep(30)
            return call_groq_llama(sys_msg, user_msg)
        print(f"API Error: {e}")
        return "[[ERROR]]"

def extract_verdict(text):
    if "[[A]]" in text: return "A"
    if "[[B]]" in text: return "B"
    if "[[C]]" in text: return "C"
    match = re.search(r'\[\[([A-C])\]\]', text)
    return match.group(1) if match else "ERROR"

def run_audit(file_in, file_out):
    os.makedirs(os.path.dirname(file_out), exist_ok=True)
    print(f"\n--- Starting Audit: {os.path.basename(file_in)} ---")
    
    with open(file_in, 'r') as f_in, open(file_out, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            cat = 'math' if data.get('category') in ['math', 'coding'] else 'general'
            config = OFFICIAL_PROMPTS.get(cat, OFFICIAL_PROMPTS['general'])
            
            # Safe reference extraction
            reference = data.get('reference', "N/A")
            if isinstance(reference, list): reference = reference[0]

            results = []
            for is_swap in [False, True]:
                ans_a = data['model_a']['responses'][0]
                ans_b = data['model_b']['responses'][0]
                if is_swap: ans_a, ans_b = ans_b, ans_a
                
                # Handling the {ref_answer_1} KeyError from your previous run
                user_content = config['prompt_template'].format(
                    question=data['prompt'],
                    answer_a=ans_a,
                    answer_b=ans_b,
                    ref_answer=reference,
                    ref_answer_1=reference
                )
                
                raw_out = call_groq_llama(config['system_prompt'], user_content)
                results.append(extract_verdict(raw_out))
                # 8B is fast, so 1s sleep is usually enough for the free tier
                time.sleep(1) 

            f_v, r_v = results[0], results[1]
            if "ERROR" in results:
                consistent = False
            else:
                consistent = (f_v == 'A' and r_v == 'B') or (f_v == 'B' and r_v == 'A') or (f_v == 'C' and r_v == 'C')

            f_out.write(json.dumps({
                "id": data['question_id'],
                "category": data.get('category', 'unknown'),
                "consistent": consistent,
                "forward": f_v,
                "reverse": r_v
            }) + "\n")
            
            print(f"ID {data['question_id']}: {'PASS' if consistent else 'FAIL'} (F:{f_v} R:{r_v})")

if __name__ == "__main__":
    if OFFICIAL_PROMPTS:
        for path in INPUT_FILES:
            bench_name = "mt_bench" if "mt_bench" in path else "vicuna_bench"
            file_name = os.path.basename(path).replace(".jsonl", "_llama8b_judge.jsonl")
            out_path = os.path.join(BASE_OUTPUT_PATH, bench_name, "judge_scores", file_name)
            run_audit(path, out_path)