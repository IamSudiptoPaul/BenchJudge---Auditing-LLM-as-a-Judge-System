import json
import os
import matplotlib.pyplot as plt
import numpy as np

def get_stats(path):
    """Extracts win rates from the judge results JSONL file."""
    wins = {"A": 0, "B": 0, "Tie": 0, "Total": 0}
    
    if not os.path.exists(path):
        print(f"Warning: File not found - {path}")
        return None
        
    with open(path, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
                w = d.get('forward')
                
                if w == "A": wins["A"] += 1
                elif w == "B": wins["B"] += 1
                elif w == "C": wins["Tie"] += 1
                
                wins["Total"] += 1
            except Exception as e:
                continue
                
    if wins["Total"] == 0: return None
    
    return [
        (wins["A"] / wins["Total"]) * 100, 
        (wins["B"] / wins["Total"]) * 100, 
        (wins["Tie"] / wins["Total"]) * 100
    ]

def plot_judge_comparison():
    BASE = "/Users/sudiptogoldfish/Documents/BenchJudge/outputs"
    
    # Define paths based on your previous directory structure
    data = {
        "MT-Bench": {
            "Llama 3.1 8B": get_stats(f"{BASE}/mt_bench/judge_scores/mtbench_answers_llama8b_judge.jsonl"),
            "Qwen": get_stats(f"{BASE}/mt_bench/judge_scores/mtbench_answers_qwen.jsonl") 
        },
        "Vicuna-Bench": {
            "Llama 3.1 8B": get_stats(f"{BASE}/vicuna_bench/judge_scores/vicuna_answers_llama8b_judge.jsonl"),
            "Qwen": get_stats(f"{BASE}/vicuna_bench/judge_scores/vicuna_answers_qwen.jsonl") 
        }
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    categories = ['Model A', 'Model B', 'Tie (C)']
    x = np.arange(len(categories))
    width = 0.35

    llama_color = "#2e5bcc" 
    qwen_color = "#ff5d55"  

    for i, (ds_name, judges) in enumerate(data.items()):
        ax = axes[i]
        
        llama_vals = judges.get("Llama 3.1 8B") or [0, 0, 0]
        qwen_vals = judges.get("Qwen") or [0, 0, 0]
        
        rects1 = ax.bar(x - width/2, llama_vals, width, label='Llama 3.1 8B', color=llama_color, edgecolor='black', alpha=0.8)
        rects2 = ax.bar(x + width/2, qwen_vals, width, label='Qwen Judge', color=qwen_color, edgecolor='black', alpha=0.8)
        
        ax.set_title(f'Win Distribution: {ds_name}', fontweight='bold', fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        if i == 0: 
            ax.set_ylabel('Win Percentage (%)', fontweight='bold', fontsize=12)
        
        ax.legend(frameon=True, loc='upper right')

        # Precise percentage labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)

    plt.suptitle('BenchJudge Meta-Evaluation: Judge Model Comparison', fontsize=18, fontweight='bold', y=1)
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    plot_judge_comparison()