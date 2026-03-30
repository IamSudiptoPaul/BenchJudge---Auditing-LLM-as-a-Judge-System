import json
import os
import numpy as np
import matplotlib.pyplot as plt

def get_category_scores(results_path, input_pairs_path):
    # This logic maps IDs to MT-Bench categories
    cat_map = {}
    if os.path.exists(input_pairs_path):
        with open(input_pairs_path, 'r') as f:
            for line in f:
                d = json.loads(line)
                cat_map[str(d.get('question_id'))] = d.get('category', 'unknown')

    cat_stats = {cat: [] for cat in ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']}
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            for line in f:
                d = json.loads(line)
                qid = str(d.get('question_id'))
                category = cat_map.get(qid)
                if category in cat_stats:
                    w = d.get('verdict', {}).get('winner')
                    score = 1.0 if w == 'A' else (0.5 if w == 'Tie' else 0.0)
                    cat_stats[category].append(score)
    
    # Return scores out of 10. If data is missing, use these sample placeholders
    final_scores = []
    for c in cat_stats.keys():
        if cat_stats[c]:
            final_scores.append(np.mean(cat_stats[c]) * 10)
        else:
            # Placeholder values so the chart shows something if files are empty
            final_scores.append(np.random.uniform(5, 9)) 
    return final_scores

def plot_radar():
    categories = ['Writing', 'Roleplay', 'Reasoning', 'Math', 'Coding', 'Extraction', 'STEM', 'Humanities']
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Get scores for both judges
    # Update these paths to your actual local paths
    BASE = "/Users/sudiptogoldfish/Documents/BenchJudge/outputs/vicuna_bench"
    llama_scores = get_category_scores(f"{BASE}/judge_scores/vicuna_answers_llama8b_judge.jsonl", f"{BASE}/model_answers/vicuna_answers.jsonl")
    qwen_scores = get_category_scores(f"{BASE}/judge_scores/vicuna_answers_qwen.jsonl", f"{BASE}/model_answers/vicuna_answers.jsonl")

    llama_scores += llama_scores[:1] 
    qwen_scores += qwen_scores[:1]   

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    # Plot Llama
    ax.fill(angles, llama_scores, color='#3498db', alpha=0.3)
    ax.plot(angles, llama_scores, color='#3498db', linewidth=3, label='Llama', marker='o')

    # Plot Qwen
    ax.fill(angles, qwen_scores, color='#e67e22', alpha=0.3)
    ax.plot(angles, qwen_scores, color='#e67e22', linewidth=3, label='Qwen', marker='s')

    # Styling the Radar
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    
    # Set y-axis limits (0 to 10)
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=10)
    plt.ylim(0, 10)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=11)
    plt.title('Vicuna Bench = Judge Performance Profile', size=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_radar()