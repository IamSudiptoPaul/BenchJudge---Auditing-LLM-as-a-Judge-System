import json
import os

RESULTS_FILES = [
    "/Users/sudiptogoldfish/Documents/BenchJudge/outputs/mt_bench/judge_scores/mtbench_answers_llama8b_judge.jsonl",
    "/Users/sudiptogoldfish/Documents/BenchJudge/outputs/mt_bench/judge_scores/mtbench_answers_qwen.jsonl",
    "/Users/sudiptogoldfish/Documents/BenchJudge/outputs/vicuna_bench/judge_scores/vicuna_answers_llama8b_judge.jsonl",
    "/Users/sudiptogoldfish/Documents/BenchJudge/outputs/vicuna_bench/judge_scores/vicuna_answers_qwen.jsonl",
]

def analyze_file(filepath):
    wins_a, wins_b, ties = 0, 0, 0
    consistent_count = 0
    total = 0
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                total += 1
                
                # 'forward' represents the first run (Llama=A, Mistral=B)
                winner = data.get("forward", "C")
                
                if winner == 'A': wins_a += 1
                elif winner == 'B': wins_b += 1
                else: ties += 1
                
                if data.get("consistent"):
                    consistent_count += 1
            except:
                continue

    return {
        "file": os.path.basename(filepath),
        "total": total,
        "consistency": (consistent_count / total * 100) if total > 0 else 0,
        "win_a": (wins_a / total * 100) if total > 0 else 0,
        "win_b": (wins_b / total * 100) if total > 0 else 0,
        "tie": (ties / total * 100) if total > 0 else 0
    }

# Header for the table
print(f"{'Judge File':<35} | {'Total':<5} | {'Consist%':<8} | {'Win A%':<7} | {'Win B%':<7} | {'Tie%'}")
print("-" * 85)

for f in RESULTS_FILES:
    stats = analyze_file(f)
    if stats:
        print(f"{stats['file']:<35} | {stats['total']:<5} | {stats['consistency']:>7.1f}% | {stats['win_a']:>6.1f}% | {stats['win_b']:>6.1f}% | {stats['tie']:>5.1f}%")