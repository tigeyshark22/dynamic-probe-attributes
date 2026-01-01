
import json
import os
import re
import argparse
import collections

CATEGORIES = ["Age", "Gender", "Socioeconomic", "Education", "Emotion", "Urgency"]
LAYERS = list(range(1, 41))

CATEGORY_CONFIG = {
    "age": {"label_map": {"child": 0, "adolescent": 1, "adult": 2, "older adult": 3}},
    "gender": {"label_map": {"male": 0, "female": 1}},
    "socioeconomic": {"label_map": {"low": 0, "middle": 1, "high": 2}},
    "education": {"label_map": {"someschool": 0, "highschool": 1, "collegemore": 2}},
    "emotion": {"label_map": {"sad": 0, "neutral emotion": 1, "happy": 2}},
    "urgency": {"label_map": {"panic": 0, "normal urgency": 1, "leisure": 2}}
}

# 0-indexed
FIRST_3_IDXS = [0, 1, 2]
LAST_3_IDXS = [3, 4, 5]
NUM_FRAGMENTS = 6

def get_change_labels(filename):
    match = re.search(r"prompt_(\d+)_(.*)_to_(.*)_conversation", filename)
    if match:
        p_id = int(match.group(1))
        l1 = match.group(2).replace("_", " ")
        l2 = match.group(3).replace("_", " ")
        return p_id, l1, l2
    return None, None, None

import numpy as np

def load_all_conversations():
    all_conversations = []
    
    for category in CATEGORIES:
        category_key = category.lower()
        if category_key == "socioeconomic status": category_key = "socioeconomic"
        
        config = CATEGORY_CONFIG.get(category_key)
        if not config: continue

        input_file = f"../data/probe_results_{category_key}_change_current.jsonl"
        if not os.path.exists(input_file): continue

        with open(input_file, 'r') as f:
            file_entries = collections.defaultdict(list)
            for line in f:
                try:
                    entry = json.loads(line)
                    file_entries[entry['file']].append(entry)
                except json.JSONDecodeError: continue
            
            for filename, entries in file_entries.items():
                entries.sort(key=lambda x: x['fragment_index'])
                if len(entries) < NUM_FRAGMENTS: continue
                
                p_id, l1, l2 = get_change_labels(filename)
                if not l1 or not l2: continue
                
                all_conversations.append({"entries": entries, "l1": l1, "l2": l2})
    return all_conversations

def main():
    conversations = load_all_conversations()
    print(f"Loaded {len(conversations)} total conversations across all categories.")

    print("-" * 140)
    header = "Layer | " + " | ".join([f"T{i+1} Mean | T{i+1} CI" for i in range(6)])
    print(header)
    print("-" * 140)

    for layer in LAYERS:
        vals_per_turn = {i: [] for i in range(6)}

        for conv in conversations:
            entries = conv["entries"]
            l1 = conv["l1"]
            l2 = conv["l2"]
            
            for idx in range(6):
                # Determine Ground Truth
                target_label = l1 if idx < 3 else l2
                
                if idx < len(entries):
                    preds = entries[idx]['predictions'].get(str(layer))
                    if preds:
                        top = max(preds, key=preds.get)
                        if top == target_label:
                            vals_per_turn[idx].append(1.0)
                        else:
                            vals_per_turn[idx].append(0.0)
        
        # Calculate stats for this layer
        row_str = f"{layer:<6}"
        for i in range(6):
            vals = vals_per_turn[i]
            if not vals:
                row_str += " | 0.0000  | 0.0000 "
                continue
                
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            n_val = len(vals)
            sem_val = std_val / np.sqrt(n_val) if n_val > 0 else 0
            ci_val = 1.96 * sem_val
            
            row_str += f" | {mean_val:<7.4f} | {ci_val:<7.4f}"
            
        print(row_str)

if __name__ == "__main__":
    main()
