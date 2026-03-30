import json, os, re, time, requests
from google import genai

GEMINI_KEY = "AIzaSyCXmxKjb6pzYjFH7gg16fMbgdLoOcFYGdg"
OLLAMA_MODEL = "qwen2.5:7b" 
OLLAMA_URL = "http://localhost:11434/api/generate"

INPUTS = [
    "outputs/vicuna_bench/model_answers/vicuna_answers.jsonl",
    "outputs/mt_bench/model_answers/mtbench_answers.jsonl"
]

client = genai.Client(api_key=GEMINI_KEY)

JUDGE_PROMPTS = {
    "general": {
        "sys": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
        "template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
    },
    "math": {
        "sys": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer, assistant A's answer, and assistant B's answer. Your job is to evaluate which assistant's answer is better. Begin your evaluation by comparing both assistants' answers with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
        "template": "[User Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer}\n[The End of Reference Answer]\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
    }
}

def call_gemini(sys, user_text):
    try:
        res = client.models.generate_content(
            model='gemini-1.5-pro', 
            contents=user_text, 
            config={'system_instruction': sys}
        )
        return res.text
    except: return "[[C]]"

def call_qwen(sys, user_text):
    prompt = f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    try:
        res = requests.post(OLLAMA_URL, json=payload).json()
        return res.get('response', '[[C]]')
    except: return "[[C]]"

def extract_verdict(text):
    match = re.search(r'\[\[([A-C])\]\]', text)
    return match.group(1) if match else "C"

def run_audit(file_in, judge_name, file_out):
    if not os.path.exists(file_in):
        print(f"Skipping: {file_in} not found.")
        return

    os.makedirs(os.path.dirname(file_out), exist_ok=True)
    print(f"\n🚀 Running {judge_name} Audit on {os.path.basename(file_in)}")
    
    with open(file_in, 'r') as f_in, open(file_out, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            cat = "math" if data.get('category') in ['math', 'coding', 'reasoning'] else "general"
            
            q = data['prompt']
            ans_llama = data['model_a']['responses'][0]
            ans_mistral = data['model_b']['responses'][0]
            ref = data.get('reference', "Evaluate based on logical correctness.")

            results = []
            # SWAP LOGIC: Forward (Llama=A, Mistral=B) | Reversed (Mistral=A, Llama=B)
            for is_swap in [False, True]:
                a_content, b_content = (ans_mistral, ans_llama) if is_swap else (ans_llama, ans_mistral)
                
                user_msg = JUDGE_PROMPTS[cat]["template"].format(
                    question=q, answer_a=a_content, answer_b=b_content, ref_answer=ref
                )
                system_msg = JUDGE_PROMPTS[cat]["sys"]

                raw = call_gemini(system_msg, user_msg) if judge_name == "gemini" else call_qwen(system_msg, user_msg)
                if judge_name == "gemini": time.sleep(4) 
                
                results.append({"verdict": extract_verdict(raw), "raw": raw})

            f_verdict, r_verdict = results[0]['verdict'], results[1]['verdict']
            consistent = (f_verdict == 'A' and r_verdict == 'B') or \
                         (f_verdict == 'B' and r_verdict == 'A') or \
                         (f_verdict == 'C' and r_verdict == 'C')

            f_out.write(json.dumps({
                "id": data['question_id'],
                "consistent": consistent,
                "forward": f_verdict,
                "reverse": r_verdict,
                "reasoning": results[0]['raw']
            }) + "\n")
            print(f"ID {data['question_id']}: Consistent={consistent}")

if __name__ == "__main__":
    for path in INPUTS:
        # Generate output paths dynamically
        base_out = path.replace("model_answers", "judge_scores").replace(".jsonl", "")
        run_audit(path, "qwen", f"{base_out}_qwen.jsonl")
        run_audit(path, "gemini", f"{base_out}_gemini_flash.jsonl")