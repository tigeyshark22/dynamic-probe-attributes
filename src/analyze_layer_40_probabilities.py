
import json
import os
import re
import glob
from collections import defaultdict

# Configuration
DATA_DIR = "../data"
TARGET_LAYER = "40"

# 0-based indices for Turns 4, 5, 6
IDX_4 = 3
IDX_5 = 4
IDX_6 = 5

def get_change_labels(filename):
    match = re.search(r"prompt_(\d+)_(.*)_to_(.*)_conversation", filename)
    if match:
        p_id = int(match.group(1))
        l1 = match.group(2).replace("_", " ")
        l2 = match.group(3).replace("_", " ")
        return p_id, l1, l2
    return None, None, None

def is_correct(predictions, target_layer, label):
    preds = predictions.get(target_layer)
    if preds:
        top = max(preds, key=preds.get)
        return top == label
    return False

def main():
    files = glob.glob(os.path.join(DATA_DIR, "probe_results_*_change_current.jsonl"))
    print(f"Found {len(files)} result files.")
    
    # Counts
    # N total samples
    # N_4_inc, N_5_inc, N_6_inc
    # N_4_inc_5_inc (4 is inc AND 5 is inc)
    # N_5_inc_6_inc (5 is inc AND 6 is inc)
    
    stats = {
        'total': 0,
        'inc_4': 0,
        'inc_5': 0,
        'inc_6': 0,
        'inc_4_and_5': 0,
        'inc_5_and_6': 0
    }
    
    for fpath in files:
        # Load and group by file
        file_entries = defaultdict(list)
        with open(fpath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    file_entries[entry['file']].append(entry)
                except json.JSONDecodeError:
                    continue
                    
        for filename, entries in file_entries.items():
            _, l1, l2 = get_change_labels(filename)
            if not l2: continue
            
            # Find entries for turns 4, 5, 6
            e4 = next((e for e in entries if e['fragment_index'] == IDX_4), None)
            e5 = next((e for e in entries if e['fragment_index'] == IDX_5), None)
            e6 = next((e for e in entries if e['fragment_index'] == IDX_6), None)
            
            if not (e4 and e5 and e6):
                continue
            
            stats['total'] += 1
            
            # For Turns 4, 5, 6, the target label is l2 (the new persona)
            correct_4 = is_correct(e4['predictions'], TARGET_LAYER, l2)
            correct_5 = is_correct(e5['predictions'], TARGET_LAYER, l2)
            correct_6 = is_correct(e6['predictions'], TARGET_LAYER, l2)
            
            inc_4 = not correct_4
            inc_5 = not correct_5
            inc_6 = not correct_6
            
            if inc_4: stats['inc_4'] += 1
            if inc_5: stats['inc_5'] += 1
            if inc_6: stats['inc_6'] += 1
            
            if inc_4 and inc_5: stats['inc_4_and_5'] += 1
            if inc_5 and inc_6: stats['inc_5_and_6'] += 1

    total = stats['total']
    if total == 0:
        print("No conversations found.")
        return

    print(f"\nAnalyzed {total} conversations.")
    print("-" * 60)
    
    # P(4 inc)
    p_4_inc = stats['inc_4'] / total
    print(f"P(4 Incorrect) = {stats['inc_4']}/{total} = {p_4_inc:.4f}")
    
    # P(5 inc)
    p_5_inc = stats['inc_5'] / total
    print(f"P(5 Incorrect) = {stats['inc_5']}/{total} = {p_5_inc:.4f}")
    
    # P(5 inc | 4 inc)
    p_5_inc_given_4_inc = 0
    if stats['inc_4'] > 0:
        p_5_inc_given_4_inc = stats['inc_4_and_5'] / stats['inc_4']
    print(f"P(5 Incorrect | 4 Incorrect) = {stats['inc_4_and_5']}/{stats['inc_4']} = {p_5_inc_given_4_inc:.4f}")
    
    print("-" * 30)
    
    # P(6 inc)
    p_6_inc = stats['inc_6'] / total
    print(f"P(6 Incorrect) = {stats['inc_6']}/{total} = {p_6_inc:.4f}")
    
    # P(6 inc | 5 inc)
    p_6_inc_given_5_inc = 0
    if stats['inc_5'] > 0:
        p_6_inc_given_5_inc = stats['inc_5_and_6'] / stats['inc_5']
    print(f"P(6 Incorrect | 5 Incorrect) = {stats['inc_5_and_6']}/{stats['inc_5']} = {p_6_inc_given_5_inc:.4f}")

if __name__ == "__main__":
    main()
