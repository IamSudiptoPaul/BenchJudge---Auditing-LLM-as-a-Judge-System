import json
import os

FILES = [
    "/Users/sudiptogoldfish/Documents/BenchJudge/outputs/mt_bench/judge_scores/mtbench_answers_llama8b_judge.jsonl",
    "/Users/sudiptogoldfish/Documents/BenchJudge/outputs/vicuna_bench/judge_scores/vicuna_answers_llama8b_judge.jsonl"
]

def analyze_bias(filepath):
    total = 0
    consistent = 0
    a_bias = 0  # Picked A both times
    b_bias = 0  # Picked B both times
    errors = 0
    
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            total += 1
            f_v, r_v = data['forward'], data['reverse']
            
            if data['consistent']:
                consistent += 1
            elif f_v == "ERROR" or r_v == "ERROR":
                errors += 1
            elif f_v == 'A' and r_v == 'A':
                a_bias += 1
            elif f_v == 'B' and r_v == 'B':
                b_bias += 1
                
    return {
        "benchmark": os.path.basename(filepath),
        "total": total,
        "consistency_rate": (consistent / total) * 100,
        "a_bias_count": a_bias,
        "b_bias_count": b_bias,
        "error_count": errors
    }

print(f"{'Benchmark':<30} | {'Consistency %':<15} | {'A-Bias':<8} | {'B-Bias':<8}")
print("-" * 75)

for f in FILES:
    stats = analyze_bias(f)
    if stats:
        print(f"{stats['benchmark']:<30} | {stats['consistency_rate']:>13.2f}% | {stats['a_bias_count']:>8} | {stats['b_bias_count']:>8}")