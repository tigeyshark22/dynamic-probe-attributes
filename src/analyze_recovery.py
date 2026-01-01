
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

# Turn mapping (1-indexed to 0-indexed)
TURN_4_IDX = 3
TURN_5_IDX = 4
TURN_6_IDX = 5
NUM_FRAGMENTS = 6

def get_change_labels(filename):
    match = re.search(r"prompt_(\d+)_(.*)_to_(.*)_conversation", filename)
    if match:
        p_id = int(match.group(1))
        l1 = match.group(2).replace("_", " ")
        l2 = match.group(3).replace("_", " ")
        return p_id, l1, l2
    return None, None, None

def load_conversations(category):
    category_key = category.lower()
    if category_key == "socioeconomic status": category_key = "socioeconomic"
    
    config = CATEGORY_CONFIG.get(category_key)
    if not config:
        return None, None

    input_file = f"../data/probe_results_{category_key}_change_current.jsonl"
    if not os.path.exists(input_file):
        return None, None

    conversations = collections.defaultdict(list)
    with open(input_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                conversations[entry['file']].append(entry)
            except json.JSONDecodeError:
                continue

    # Filter and sort
    valid_conversations = []
    for filename, entries in conversations.items():
        entries.sort(key=lambda x: x['fragment_index'])
        if len(entries) < NUM_FRAGMENTS:
            continue

        p_id, l1, l2 = get_change_labels(filename)
        if not l1 or not l2:
            continue
        valid_conversations.append({"entries": entries, "l2": l2})
        
    return valid_conversations, category_key

def compute_layer_stats(conversations, layer):
    total_conversations = 0
    incorrect_turn_4 = 0
    recovered = 0

    for item in conversations:
        entries = item["entries"]
        l2 = item["l2"]
        
        try:
            pred_turn_4 = entries[TURN_4_IDX]['predictions'].get(str(layer))
            pred_turn_5 = entries[TURN_5_IDX]['predictions'].get(str(layer))
            pred_turn_6 = entries[TURN_6_IDX]['predictions'].get(str(layer))
        except (IndexError, KeyError):
            continue

        if not pred_turn_4 or not pred_turn_5 or not pred_turn_6:
            continue
            
        total_conversations += 1

        top_4 = max(pred_turn_4, key=pred_turn_4.get)
        top_5 = max(pred_turn_5, key=pred_turn_5.get)
        top_6 = max(pred_turn_6, key=pred_turn_6.get)

        is_correct_4 = (top_4 == l2)
        is_correct_5 = (top_5 == l2)
        is_correct_6 = (top_6 == l2)

        if not is_correct_4:
            incorrect_turn_4 += 1
            if is_correct_5 or is_correct_6:
                recovered += 1

    return {
        "total": total_conversations,
        "incorrect_turn_4": incorrect_turn_4,
        "recovered": recovered
    }

def main():
    print(f"{'Category':<15} | {'Layer':<5} | {'P(Incorrect T4)':<18} | {'P(Recovery)':<15}")
    print("-" * 65)
    
    for category in CATEGORIES:
        conversations, _ = load_conversations(category)
        if not conversations:
            print(f"{category:<15} | {'ALL':<5} | {'DATA MISSING':<18} | {'':<15}")
            print("-" * 65)
            continue
            
        for layer in LAYERS:
            stats = compute_layer_stats(conversations, layer)
            
            total = stats['total']
            inc_t4 = stats['incorrect_turn_4']
            rec = stats['recovered']
            
            p_inc_t4 = inc_t4 / total if total > 0 else 0.0
            p_rec = rec / inc_t4 if inc_t4 > 0 else 0.0
            
            print(f"{category:<15} | {layer:<5} | {p_inc_t4:<18.4f} | {p_rec:<15.4f}")
        print("-" * 65)

if __name__ == "__main__":
    main()
