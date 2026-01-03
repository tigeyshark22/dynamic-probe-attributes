
import json
import os
import re
import glob
from collections import defaultdict

# Configuration
DATA_DIR = "../data"
TARGET_LAYER = "40"

# Turn Indices (0-based)
CHANGE_TURN_IDX = 3  # Turn 4
NEXT_TURN_1_IDX = 4  # Turn 5
NEXT_TURN_2_IDX = 5  # Turn 6

def get_change_labels(filename):
    match = re.search(r"prompt_(\d+)_(.*)_to_(.*)_conversation", filename)
    if match:
        p_id = int(match.group(1))
        l1 = match.group(2).replace("_", " ")
        l2 = match.group(3).replace("_", " ")
        return p_id, l1, l2
    return None, None, None

def load_and_analyze():
    files = glob.glob(os.path.join(DATA_DIR, "probe_results_*_change_current.jsonl"))
    print(f"Found {len(files)} result files.")
    
    # Counts
    # Structure: [T4_Status][Next_Two_Status]
    # T4_Status: 'Correct', 'Incorrect'
    # Next_Two_Status: 'Both_Correct', 'Both_Incorrect', 'Mixed'
    counts = {
        'Correct': {'Both_Correct': 0, 'Both_Incorrect': 0, 'Mixed': 0},
        'Incorrect': {'Both_Correct': 0, 'Both_Incorrect': 0, 'Mixed': 0}
    }
    
    total_convs = 0
    
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
            
            # We need indices 3, 4, 5
            t4_entry = next((e for e in entries if e['fragment_index'] == CHANGE_TURN_IDX), None)
            t5_entry = next((e for e in entries if e['fragment_index'] == NEXT_TURN_1_IDX), None)
            t6_entry = next((e for e in entries if e['fragment_index'] == NEXT_TURN_2_IDX), None)
            
            if not (t4_entry and t5_entry and t6_entry):
                continue
                
            total_convs += 1
            
            # Check T4 (Turn 4) - Target is l2
            # Note: At Turn 4, the persona HAS switched. Target is l2.
            pred4 = t4_entry['predictions'].get(TARGET_LAYER)
            is_4_correct = False
            if pred4:
                top4 = max(pred4, key=pred4.get)
                if top4 == l2:
                    is_4_correct = True
            
            # Check T5 (Turn 5)
            pred5 = t5_entry['predictions'].get(TARGET_LAYER)
            is_5_correct = False
            if pred5:
                top5 = max(pred5, key=pred5.get)
                if top5 == l2:
                    is_5_correct = True
                    
            # Check T6 (Turn 6)
            pred6 = t6_entry['predictions'].get(TARGET_LAYER)
            is_6_correct = False
            if pred6:
                top6 = max(pred6, key=pred6.get)
                if top6 == l2:
                    is_6_correct = True
            
            # Categorize
            status_4 = 'Correct' if is_4_correct else 'Incorrect'
            
            if is_5_correct and is_6_correct:
                status_next = 'Both_Correct'
            elif (not is_5_correct) and (not is_6_correct):
                status_next = 'Both_Incorrect'
            else:
                status_next = 'Mixed'
                
            counts[status_4][status_next] += 1

    print(f"\nAnalyzed {total_convs} conversations.")
    print("-" * 60)
    print(f"Layer {TARGET_LAYER} Recovery Probability Square")
    print("-" * 60)
    
    # Calculate Probabilities
    # P(Next | 4) = Count(4 & Next) / Count(4)
    
    for t4_stat in ['Correct', 'Incorrect']:
        total_t4 = sum(counts[t4_stat].values())
        if total_t4 == 0:
            print(f"T4 {t4_stat}: No samples.")
            continue
            
        p_both_correct = counts[t4_stat]['Both_Correct'] / total_t4
        p_both_incorrect = counts[t4_stat]['Both_Incorrect'] / total_t4
        p_mixed = counts[t4_stat]['Mixed'] / total_t4
        
        print(f"Given Turn 4 was {t4_stat.upper()} (N={total_t4}):")
        print(f"  P(5 & 6 Correct)   = {p_both_correct:.2%}")
        print(f"  P(5 & 6 Incorrect) = {p_both_incorrect:.2%}")
        print(f"  (Mixed/Other       = {p_mixed:.2%})")
        print()
        
    # Formatted Square
    print("\n[ Conditional Matrix ]")
    print(f"{'':<20} | {'5&6 Correct':<15} | {'5&6 Incorrect':<15}")
    print("-" * 56)
    
    row1_n = sum(counts['Correct'].values())
    r1_p1 = counts['Correct']['Both_Correct'] / row1_n if row1_n else 0
    r1_p2 = counts['Correct']['Both_Incorrect'] / row1_n if row1_n else 0
    
    print(f"{'4 Correct':<20} | {r1_p1:.4f}          | {r1_p2:.4f}")
    
    row2_n = sum(counts['Incorrect'].values())
    r2_p1 = counts['Incorrect']['Both_Correct'] / row2_n if row2_n else 0
    r2_p2 = counts['Incorrect']['Both_Incorrect'] / row2_n if row2_n else 0
    
    print(f"{'4 Incorrect':<20} | {r2_p1:.4f}          | {r2_p2:.4f}")


if __name__ == "__main__":
    load_and_analyze()
