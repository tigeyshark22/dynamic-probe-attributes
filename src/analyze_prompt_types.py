
import json
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
DATA_DIR = "../data"
LAYERS_TO_AVERAGE = list(range(35, 41))

# Definitions
PROMPT_TYPES = {
    1: {"type": "Explicit", "pickup": "Pickup"},
    2: {"type": "Explicit", "pickup": "No Pickup"},
    3: {"type": "Implicit", "pickup": "Pickup"},
    4: {"type": "Implicit", "pickup": "No Pickup"}
}

def get_metadata_from_filename(filename):
    # format: ..._prompt_X_...
    match = re.search(r"prompt_(\d+)_", filename)
    if match:
        p_id = int(match.group(1))
        return PROMPT_TYPES.get(p_id)
    return None

def get_change_labels(filename):
    match = re.search(r"prompt_(\d+)_(.*)_to_(.*)_conversation", filename)
    if match:
        p_id = int(match.group(1))
        l1 = match.group(2).replace("_", " ")
        l2 = match.group(3).replace("_", " ")
        return p_id, l1, l2
    return None, None, None

def load_data():
    files = glob.glob(os.path.join(DATA_DIR, "probe_results_*_change_current.jsonl"))
    
    # type_probs split by L1/L2 - Explicit vs Implicit
    # Filter 1: Pickup
    type_probs_pickup_l1 = {"Explicit": {t: [] for t in range(6)}, "Implicit": {t: [] for t in range(6)}}
    type_probs_pickup_l2 = {"Explicit": {t: [] for t in range(6)}, "Implicit": {t: [] for t in range(6)}}
    # Filter 2: No Pickup
    type_probs_nopickup_l1 = {"Explicit": {t: [] for t in range(6)}, "Implicit": {t: [] for t in range(6)}}
    type_probs_nopickup_l2 = {"Explicit": {t: [] for t in range(6)}, "Implicit": {t: [] for t in range(6)}}
    
    # pickup_probs split by L1/L2 - Pickup vs No Pickup
    # Filter 1: Explicit
    pickup_probs_explicit_l1 = {"Pickup": {t: [] for t in range(6)}, "No Pickup": {t: [] for t in range(6)}}
    pickup_probs_explicit_l2 = {"Pickup": {t: [] for t in range(6)}, "No Pickup": {t: [] for t in range(6)}}
    # Filter 2: Implicit
    pickup_probs_implicit_l1 = {"Pickup": {t: [] for t in range(6)}, "No Pickup": {t: [] for t in range(6)}}
    pickup_probs_implicit_l2 = {"Pickup": {t: [] for t in range(6)}, "No Pickup": {t: [] for t in range(6)}}
    
    print(f"Found {len(files)} result files.")
    
    for fpath in files:
        print(f"Processing {os.path.basename(fpath)}...")
        
        file_entries = defaultdict(list)
        with open(fpath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    file_entries[entry['file']].append(entry)
                except json.JSONDecodeError:
                    continue
        
        for filename, entries in file_entries.items():
            entries.sort(key=lambda x: x['fragment_index'])
            
            metadata = get_metadata_from_filename(filename)
            if not metadata:
                continue
                
            p_type = metadata["type"]
            p_pickup = metadata["pickup"]
            
            p_id, l1, l2 = get_change_labels(filename)
            if not l1 or not l2: continue
            
            for idx in range(6):
                target_label = l1 if idx < 3 else l2
                
                entry = next((e for e in entries if e['fragment_index'] == idx), None)
                
                if entry:
                    l1_prob_sum = 0
                    l2_prob_sum = 0
                    layer_valid_count = 0
                    
                    predictions = entry['predictions']
                    
                    for layer in LAYERS_TO_AVERAGE:
                        preds = predictions.get(str(layer))
                        if preds:
                            l1_prob_sum += preds.get(l1, 0.0)
                            l2_prob_sum += preds.get(l2, 0.0)
                            layer_valid_count += 1
                    
                    if layer_valid_count > 0:
                        avg_l1 = l1_prob_sum / layer_valid_count
                        avg_l2 = l2_prob_sum / layer_valid_count
                        
                        # 1. Populate Explicit vs Implicit
                        if p_pickup == "Pickup":
                            type_probs_pickup_l1[p_type][idx].append(avg_l1)
                            type_probs_pickup_l2[p_type][idx].append(avg_l2)
                        elif p_pickup == "No Pickup":
                            type_probs_nopickup_l1[p_type][idx].append(avg_l1)
                            type_probs_nopickup_l2[p_type][idx].append(avg_l2)
                        
                        # 2. Populate Pickup vs No Pickup
                        if p_type == "Explicit":
                            pickup_probs_explicit_l1[p_pickup][idx].append(avg_l1)
                            pickup_probs_explicit_l2[p_pickup][idx].append(avg_l2)
                        elif p_type == "Implicit":
                            pickup_probs_implicit_l1[p_pickup][idx].append(avg_l1)
                            pickup_probs_implicit_l2[p_pickup][idx].append(avg_l2)

    return (
        type_probs_pickup_l1, type_probs_pickup_l2, 
        type_probs_nopickup_l1, type_probs_nopickup_l2,
        pickup_probs_explicit_l1, pickup_probs_explicit_l2,
        pickup_probs_implicit_l1, pickup_probs_implicit_l2
    )

def plot_accuracy(data_dict, title, filename, colors, markers):
    pass # Helper not strictly needed for the new task, but keeping placeholder to avoid syntax error if called

def plot_probs_l1_l2(data_l1, data_l2, title, filename, group_labels):
    # group_labels: [Group1_Name (Solid), Group2_Name (Dashed)]
    plt.figure(figsize=(12, 7))
    turns = np.array([1, 2, 3, 4, 5, 6])
    
    g1_name = group_labels[0]
    g2_name = group_labels[1]
    
    # Define styles based on reference
    # L1 = Green, L2 = Blue
    # Group 1 = Solid, Circle
    # Group 2 = Dashed, Square
    
    # Plot Group 1 (Solid)
    if g1_name in data_l1:
        # G1 - L1
        vals = data_l1[g1_name]
        means = [np.mean(vals[t]) if vals[t] else 0.0 for t in range(6)]
        plt.plot(turns, means, label=f"{g1_name} - First half correct label", color='green', linestyle='-', marker='o', linewidth=2)
        
        # G1 - L2
        vals = data_l2[g1_name]
        means = [np.mean(vals[t]) if vals[t] else 0.0 for t in range(6)]
        plt.plot(turns, means, label=f"{g1_name} - Second half correct label", color='blue', linestyle='-', marker='o', linewidth=2)

    # Plot Group 2 (Dashed)
    if g2_name in data_l1:
        # G2 - L1
        vals = data_l1[g2_name]
        means = [np.mean(vals[t]) if vals[t] else 0.0 for t in range(6)]
        plt.plot(turns, means, label=f"{g2_name} - First half correct label", color='green', linestyle='--', marker='s', linewidth=2, alpha=0.7)
        
        # G2 - L2
        vals = data_l2[g2_name]
        means = [np.mean(vals[t]) if vals[t] else 0.0 for t in range(6)]
        plt.plot(turns, means, label=f"{g2_name} - Second half correct label", color='blue', linestyle='--', marker='s', linewidth=2, alpha=0.7)

    # Shading and layout
    plt.axvspan(0.5, 3.5, color='green', alpha=0.05)
    plt.axvspan(3.5, 6.5, color='blue', alpha=0.05)
    plt.axvline(x=3.5, color='black', linestyle='-', alpha=0.3, label="Switch Point")

    plt.xlabel('Turn Number')
    plt.ylabel(f'Average Probe Output (Layers {min(LAYERS_TO_AVERAGE)}-{max(LAYERS_TO_AVERAGE)})')
    plt.title(title)
    
    # Move legend inside
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    plt.xlim(0.5, 6.5) # Match reference xlim
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

def main():
    (
        type_probs_pickup_l1, type_probs_pickup_l2, 
        type_probs_nopickup_l1, type_probs_nopickup_l2,
        pickup_probs_explicit_l1, pickup_probs_explicit_l2,
        pickup_probs_implicit_l1, pickup_probs_implicit_l2
    ) = load_data()
    
    # 1. Explicit vs Implicit (Pickup Only) -- EXISTING
    plot_probs_l1_l2(
        type_probs_pickup_l1, type_probs_pickup_l2,
        "Probe Output: Explicit vs Implicit (Pickup Only)", 
        "graphs/probe_output_vs_turn_explicit_implicit_pickup.png",
        ["Explicit", "Implicit"]
    )

    # 2. Explicit vs Implicit (No Pickup Only) -- NEW
    plot_probs_l1_l2(
        type_probs_nopickup_l1, type_probs_nopickup_l2,
        "Probe Output: Explicit vs Implicit (No Pickup Only)", 
        "graphs/probe_output_vs_turn_explicit_implicit_nopickup.png",
        ["Explicit", "Implicit"]
    )
    
    # 3. Pickup vs No Pickup (Explicit Only) -- EXISTING
    plot_probs_l1_l2(
        pickup_probs_explicit_l1, pickup_probs_explicit_l2,
        "Probe Output: Pickup vs No Pickup (Explicit Only)", 
        "graphs/probe_output_vs_turn_pickup_nopickup_explicit.png",
        ["Pickup", "No Pickup"]
    )

    # 4. Pickup vs No Pickup (Implicit Only) -- NEW
    plot_probs_l1_l2(
        pickup_probs_implicit_l1, pickup_probs_implicit_l2,
        "Probe Output: Pickup vs No Pickup (Implicit Only)", 
        "graphs/probe_output_vs_turn_pickup_nopickup_implicit.png",
        ["Pickup", "No Pickup"]
    )

if __name__ == "__main__":
    main()

