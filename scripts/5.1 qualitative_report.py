import json
import random
import os

def load_jsonl_dict(path):
    data = {}
    if not os.path.exists(path):
        print(f"Warning: Path not found: {path}")
        return data
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                qid = str(item.get("question_id") or item.get("id"))
                data[qid] = item
            except:
                continue
    return data

def export_full_text():
    BASE = "/Users/sudiptogoldfish/Documents/BenchJudge/outputs"
    full_report = "BENCHJUDGE QUALITATIVE AUDIT REPORT\n\n"

    file_map = {
        "mt_bench": "mtbench_answers",
        "vicuna_bench": "vicuna_answers"
    }

    judges = ["llama8b_judge", "qwen"]

    for ds_folder, file_prefix in file_map.items():
        input_p = f"{BASE}/{ds_folder}/model_answers/{file_prefix}.jsonl"
        inputs = load_jsonl_dict(input_p)

        for judge_suffix in judges:
            score_p = f"{BASE}/{ds_folder}/judge_scores/{file_prefix}_{judge_suffix}.jsonl"
            scores = load_jsonl_dict(score_p)

            if not inputs or not scores:
                continue

            common_ids = list(set(inputs.keys()) & set(scores.keys()))
            if not common_ids:
                continue
            
            selected_id = random.choice(common_ids)
            row = inputs[selected_id]
            audit = scores[selected_id]

            q_raw = row.get('prompt') or row.get('question_turns') or row.get('instruction')
            q_text = q_raw[0] if isinstance(q_raw, list) else q_raw

            try:
                a_full = row['model_a']['responses'][0]
                b_full = row['model_b']['responses'][0]
            except:
                a_full = "Error extracting A"
                b_full = "Error extracting B"

            f_v = audit.get('forward', 'N/A')
            r_v = audit.get('reverse', 'N/A')
            consistent = audit.get('consistent', False)
            status = "CONSISTENT" if consistent else "BIASED"

            full_report += f"DATASET: {ds_folder.upper()}\n"
            full_report += f"JUDGE: {judge_suffix.upper()}\n"
            full_report += f"ID: {selected_id} | STATUS: {status}\n"
            full_report += f"PROMPT: {q_text}\n\n"
            full_report += f"RESPONSE A: {a_full[:500]}...\n\n"
            full_report += f"RESPONSE B: {b_full[:500]}...\n\n"
            full_report += f"VERDICTS: Forward={f_v}, Reverse={r_v}\n"
            full_report += f"CATEGORY: {audit.get('category', 'N/A')}\n"
            full_report += "\n" + "-"*30 + "\n\n"

    with open("qualitative_audit_report.txt", "w", encoding='utf-8') as f:
        f.write(full_report)
    print("Report generated: qualitative_audit_report.txt")

if __name__ == "__main__":
    export_full_text()