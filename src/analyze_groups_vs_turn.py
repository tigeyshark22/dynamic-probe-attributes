
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import defaultdict

# Configuration
DATA_DIR = "../data"
OUTPUT_PLOT = "graphs/accuracy_vs_turn_by_group.png"
LAYERS_TO_AVERAGE = list(range(35, 41)) # Average layers 35-40

GROUPS = {
    "Demographic": ["age", "gender", "socioeconomic", "education"],
    "Transient": ["emotion", "urgency"]
}

def get_category_from_filename(filename):
    # filename format: probe_results_{category}_change_current.jsonl
    basename = os.path.basename(filename)
    # Remove prefix and suffix
    if basename.startswith("probe_results_") and basename.endswith("_change_current.jsonl"):
        cat_part = basename.replace("probe_results_", "").replace("_change_current.jsonl", "")
        return cat_part
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
    
    # Structure: group -> turn_idx (0-5) -> list of data
    group_accs = {g: {t: [] for t in range(6)} for g in GROUPS}
    group_probs_l1 = {g: {t: [] for t in range(6)} for g in GROUPS}
    group_probs_l2 = {g: {t: [] for t in range(6)} for g in GROUPS}
    
    print(f"Found {len(files)} result files.")
    
    for fpath in files:
        category = get_category_from_filename(fpath)
        if not category:
            continue
            
        # Determine Group
        target_group = None
        for g_name, cats in GROUPS.items():
            if category in cats:
                target_group = g_name
                break
        
        if not target_group:
            print(f"Skipping category: {category}")
            continue
            
        print(f"Processing {category} -> {target_group}...")
        
        # Load and group by file
        file_entries = defaultdict(list)
        with open(fpath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    file_entries[entry['file']].append(entry)
                except json.JSONDecodeError:
                    continue
                    
        # Process each conversation
        for filename, entries in file_entries.items():
            # Sort by fragment index to align with turns
            entries.sort(key=lambda x: x['fragment_index'])
            
            # Get Ground Truth Labels from filename
            _, l1, l2 = get_change_labels(filename)
            if not l1 or not l2: 
                continue
                
            for idx in range(6):
                target_label = l1 if idx < 3 else l2
                
                entry = next((e for e in entries if e['fragment_index'] == idx), None)
                
                if entry:
                    layer_correct_count = 0
                    layer_valid_count = 0
                    
                    l1_prob_sum = 0
                    l2_prob_sum = 0
                    
                    predictions = entry['predictions']
                    
                    for layer in LAYERS_TO_AVERAGE:
                        preds = predictions.get(str(layer))
                        if preds:
                            top = max(preds, key=preds.get)
                            if top == target_label:
                                layer_correct_count += 1
                                
                            l1_prob_sum += preds.get(l1, 0.0)
                            l2_prob_sum += preds.get(l2, 0.0)
                            
                            layer_valid_count += 1
                    
                    if layer_valid_count > 0:
                        avg_acc = layer_correct_count / layer_valid_count
                        avg_l1 = l1_prob_sum / layer_valid_count
                        avg_l2 = l2_prob_sum / layer_valid_count
                        
                        group_accs[target_group][idx].append(avg_acc)
                        group_probs_l1[target_group][idx].append(avg_l1)
                        group_probs_l2[target_group][idx].append(avg_l2)

    return group_accs, group_probs_l1, group_probs_l2

def plot_accuracy(group_data, title, ylabel, output_file):
    # Old logic for accuracy
    plt.figure(figsize=(10, 6))
    turns = np.array([1, 2, 3, 4, 5, 6])
    colors = {"Demographic": "blue", "Transient": "red"}
    markers = {"Demographic": "o", "Transient": "s"}
    
    for group, data in group_data.items():
        means = []
        cis = []
        for t in range(6):
            vals = data[t]
            if not vals: means.append(0); cis.append(0); continue
            mean = np.mean(vals); sem = np.std(vals)/np.sqrt(len(vals)); ci=1.96*sem
            means.append(mean); cis.append(ci)
        means = np.array(means); cis = np.array(cis)
        plt.plot(turns, means, label=f"{group} Categories", color=colors[group], marker=markers[group], linewidth=2)
        plt.fill_between(turns, means - cis, means + cis, color=colors[group], alpha=0.2)
    plt.xlabel('Turn Number'); plt.ylabel(ylabel); plt.title(title)
    plt.axvline(x=3.5, color='gray', linestyle='--', alpha=0.7, label='Persona Change')
    plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(0, 1.0)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

def plot_probs_l1_l2(data_l1, data_l2, title, ylabel, output_file):
    plt.figure(figsize=(12, 7))
    turns = np.array([1, 2, 3, 4, 5, 6])
    
    # Hardcoded groups for styling consistency
    g1_name = "Demographic" # Solid
    g2_name = "Transient"   # Dashed
    
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
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Move legend inside
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    plt.xlim(0.5, 6.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

def main():
    accs, probs_l1, probs_l2 = load_data()
    
    # Plot Accuracy
    plot_accuracy(
        accs, 
        'Probe Accuracy vs Turn Number: Demographic vs Transient Groups',
        f'Average Accuracy (Layers {min(LAYERS_TO_AVERAGE)}-{max(LAYERS_TO_AVERAGE)})',
        "graphs/accuracy_vs_turn_by_group.png"
    )
    
    # Plot Probe Output (L1 vs L2)
    # Using the new styled function
    plot_probs_l1_l2(
        probs_l1, probs_l2,
        'Probe Output vs Turn Number: Demographic vs Transient Groups',
        f'Average Probe Output (Layers {min(LAYERS_TO_AVERAGE)}-{max(LAYERS_TO_AVERAGE)})',
        "graphs/probe_output_vs_turn_by_group.png"
    )

if __name__ == "__main__":
    main()
