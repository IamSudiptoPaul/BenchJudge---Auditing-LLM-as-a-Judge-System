import json
import os

def analyze_results(file_path):
    total = 0
    consistent_count = 0
    picked_a_count = 0  # How many times the judge chose Position A
    picked_b_count = 0  # How many times the judge chose Position B
    picked_tie_count = 0 # How many times the judge chose [[C]]
    
    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                total += 1
                
                # Check consistency
                if data.get('consistent'):
                    consistent_count += 1
                
                # Use the full key names from the JSONL
                fwd = data.get('forward')
                rev = data.get('reverse')
                
                # Track position choices across both runs
                for verdict in [fwd, rev]:
                    if verdict == 'A': picked_a_count += 1
                    elif verdict == 'B': picked_b_count += 1
                    else: picked_tie_count += 1
            except Exception as e:
                continue
                
    if total == 0:
        print(f"No data found in {file_path}")
        return

    # Metrics calculation
    consistency_rate = (consistent_count / total) * 100
    # Position bias: percentage of total choices that were for Position A
    a_bias = (picked_a_count / (total * 2)) * 100
    b_bias = (picked_b_count / (total * 2)) * 100
    
    print(f"\n" + "="*40)
    print(f"RESULTS FOR: {os.path.basename(file_path)}")
    print(f"="*40)
    print(f"Total Questions evaluated: {total}")
    print(f"Consistency Rate:         {consistency_rate:.2f}%")
    print(f"Position A Selection:     {a_bias:.2f}%")
    print(f"Position B Selection:     {b_bias:.2f}%")
    print(f"Tie/Invalid Selection:    {(picked_tie_count/(total*2))*100:.2f}%")
    print("-" * 40)
    
    if a_bias > 70:
        print("🚩 High Position A Bias detected.")
    elif consistency_rate < 50:
        print("🚩 Low Reliability: Judge is likely guessing or order-dependent.")
    else:
        print("✅ Judge shows stable reliability.")

# Run it
analyze_results("outputs/mt_bench/judge_scores/mtbench_answers_llama8b_judge.jsonl")
analyze_results("outputs/mt_bench/judge_scores/mtbench_answers_qwen.jsonl")