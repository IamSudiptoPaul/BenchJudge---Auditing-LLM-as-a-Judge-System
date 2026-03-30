import json
import random
import os
import matplotlib.pyplot as plt

def load_jsonl_dict(path):
    data = {}
    if not os.path.exists(path): return data
    with open(path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                data[item.get("question_id") or item.get("id")] = item
            except: continue
    return data

def get_status(v1, v2):
    v1_w = v1.get('winner') if isinstance(v1, dict) else "FAIL"
    v2_w = v2.get('winner') if isinstance(v2, dict) else "FAIL"
    if v1_w == "FAIL" or v2_w == "FAIL": return "FAILED", "#eeeeee"
    if (v1_w == 'A' and v2_w == 'B') or (v1_w == 'B' and v2_w == 'A') or (v1_w == 'Tie' and v2_w == 'Tie'):
        return "CONSISTENT", "#ccffcc"
    return "BIASED", "#ffcccc"

def generate_comprehensive_audit():
    BASE = "/Users/sudiptogoldfish/Documents/BenchJudge A Meta Evaluation Framework for Auditing LLM as a Judge Systems/outputs"
    
    # 1. Choose Dataset
    dataset_choice = random.choice(["mt_bench", "chatbotarena"])
    print(f"Auditing Dataset: {dataset_choice}")

    # 2. Set Paths
    input_path = f"{BASE}/{dataset_choice}/model_answers/judge_input_pairs_{dataset_choice}.jsonl"
    l_orig_p = f"{BASE}/{dataset_choice}/judge_scores/{dataset_choice}results_llama3_1_8b.jsonl"
    l_swap_p = f"{BASE}/{dataset_choice}/judge_scores/swapped_{dataset_choice}_llama3_1_8b.jsonl"
    p_orig_p = f"{BASE}/{dataset_choice}/judge_scores/{dataset_choice}results_phi3_5.jsonl"
    p_swap_p = f"{BASE}/{dataset_choice}/judge_scores/swapped_{dataset_choice}_phi3_5.jsonl"

    # 3. Load Data
    inputs = load_jsonl_dict(input_path)
    l_orig = load_jsonl_dict(l_orig_p)
    l_swap = load_jsonl_dict(l_swap_p)
    p_orig = load_jsonl_dict(p_orig_p)
    p_swap = load_jsonl_dict(p_swap_p)

    # 4. Pick Random ID present in inputs
    selected_id = random.choice(list(inputs.keys()))
    row = inputs[selected_id]

    # 5. Extract Details
    question = row.get('question_turns', [row.get('prompt', 'N/A')])[0]
    ans_a = row['model_a']['responses'][0] if 'model_a' in row else "N/A"
    ans_b = row['model_b']['responses'][0] if 'model_b' in row else "N/A"

    # 6. Get Judge Results
    l_res1 = l_orig.get(selected_id, {}).get('verdict', {})
    l_res2 = l_swap.get(selected_id, {}).get('verdict', {})
    p_res1 = p_orig.get(selected_id, {}).get('verdict', {})
    p_res2 = p_swap.get(selected_id, {}).get('verdict', {})

    l_stat, l_col = get_status(l_res1, l_res2)
    p_stat, p_col = get_status(p_res1, p_res2)

    # 7. Visualize
    fig = plt.figure(figsize=(14, 10))
    plt.axis('off')
    plt.title(f"BenchJudge Detailed Audit | Dataset: {dataset_choice} | ID: {selected_id}", fontsize=16, fontweight='bold')

    # Content Boxes
    text_props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0, 0.9, f"QUESTION:\n{question[:300]}...", transform=plt.gca().transAxes, verticalalignment='top', bbox=text_props, fontsize=10)
    plt.text(0, 0.75, f"MODEL A RESPONSE:\n{ans_a[:400]}...", transform=plt.gca().transAxes, verticalalignment='top', bbox=text_props, fontsize=9, color='blue')
    plt.text(0, 0.55, f"MODEL B RESPONSE:\n{ans_b[:400]}...", transform=plt.gca().transAxes, verticalalignment='top', bbox=text_props, fontsize=9, color='purple')

    # Verdict Table
    table_data = [
        ["Judge Model", "Original Winner", "Swapped Winner", "Verdict Status"],
        ["Llama 3.1 8B", l_res1.get('winner', 'FAIL'), l_res2.get('winner', 'FAIL'), l_stat],
        ["Phi 3.5", p_res1.get('winner', 'FAIL'), p_res2.get('winner', 'FAIL'), p_stat]
    ]
    
    table = plt.table(cellText=table_data, loc='bottom', cellLoc='center', colWidths=[0.2]*4)
    table.scale(1, 2.5)
    table[(1, 3)].set_facecolor(l_col)
    table[(2, 3)].set_facecolor(p_col)

    plt.savefig('detailed_audit_output.png', dpi=300, bbox_inches='tight')
    print(f"Generated comprehensive audit for {dataset_choice} ID {selected_id}")

if __name__ == "__main__":
    generate_comprehensive_audit()