
import json
import os
import glob
import numpy as np
import collections
import matplotlib.pyplot as plt
import argparse
import re

# Configuration map
CATEGORY_CONFIG = {
    "age": {
        "prefix": "age",
        "label_map": {"child": 0, "adolescent": 1, "adult": 2, "older adult": 3},
        "chance": 0.25
    },
    "gender": {
        "prefix": "gender",
        "label_map": {"male": 0, "female": 1},
        "chance": 0.5
    },
    "socioeconomic": {
        "prefix": "socioeco",
        "label_map": {"low": 0, "middle": 1, "high": 2},
        "chance": 0.333
    },
    "education": {
        "prefix": "education",
        "label_map": {"someschool": 0, "highschool": 1, "collegemore": 2},
        "chance": 0.333
    },
    "emotion": {
        "prefix": "emotion",
        "label_map": {"sad": 0, "neutral emotion": 1, "happy": 2},
        "chance": 0.333
    },
    "urgency": {
        "prefix": "urgency",
        "label_map": {"panic": 0, "normal urgency": 1, "leisure": 2},
        "chance": 0.333
    }
}

# Fixed number of fragments as per user specification
NUM_FRAGMENTS = 6
# Focusing on deeper layers as per previous analyses
LAYERS = [36, 37, 38, 39, 40]

def get_change_labels(filename):
    """
    Extracts prompt_id, l1, l2 from filename.
    Format: ..._prompt_{id}_{label1}_to_{label2}_conversation_{id}.txt
    """
    match = re.search(r"prompt_(\d+)_(.*)_to_(.*)_conversation", filename)
    if match:
        p_id = int(match.group(1))
        l1 = match.group(2)
        l2 = match.group(3)
        
        l1 = l1.replace("_", " ")
        l2 = l2.replace("_", " ")
        
        return p_id, l1, l2
    return None, None, None

def analyze_change_dynamics(category, current_probe=False):
    config = CATEGORY_CONFIG.get(category)
    if not config:
        print(f"Error: Unknown category '{category}'")
        return
    
    suffix = "_current" if current_probe else ""
    input_file = f"../data/probe_results_{category}_change{suffix}.jsonl"
    random_chance = config["chance"]
    
    print(f"Analyzing Change Dynamics for Category: {category}")
    os.makedirs("graphs", exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return
    
    # Define Groups
    groups = {
        "All": [1, 2, 3, 4],
        "Pickup (Prompts 1,3)": [1, 3],
        "No Pickup (Prompts 2,4)": [2, 4],
        "Explicit Switch (Prompts 1,2)": [1, 2],
        "Implicit Switch (Prompts 3,4)": [3, 4]
    }
    
    # Structure: group -> metric -> layer -> fragment_index -> list of values
    # Metrics: 'prob_l1', 'prob_l2', 'prob_other'
    # Structure: group -> metric -> layer -> fragment_index -> list of values
    # Metrics: 'prob_l1', 'prob_l2', 'prob_other', 'acc'
    metrics = ['prob_l1', 'prob_l2', 'prob_other', 'acc']
    grouped_data = {g: {m: collections.defaultdict(lambda: collections.defaultdict(list)) for m in metrics} 
                    for g in groups}

    conversations = collections.defaultdict(list)

    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                conversations[entry['file']].append(entry)
            except json.JSONDecodeError:
                continue

    print(f"Processing {len(conversations)} conversations...")
    
    valid_conv_count = 0
    
    for filename, entries in conversations.items():
        entries.sort(key=lambda x: x['fragment_index'])
        
        p_id, l1, l2 = get_change_labels(filename)
        
        if not l1 or not l2:
            continue
        if l1 not in config["label_map"] or l2 not in config["label_map"]:
            continue
            
        valid_conv_count += 1
        
        # Identify which groups this conversation belongs to
        relevant_groups = []
        for g_name, p_ids in groups.items():
            if p_id in p_ids:
                relevant_groups.append(g_name)
        
         # Split point logic (Turn 3 is last of first persona, Turn 4 is first of second persona)
        split_idx = 3 
        
        for i, entry in enumerate(entries):
            frag_idx = entry['fragment_index']
            if frag_idx >= NUM_FRAGMENTS:
                continue
            
            # Determine Ground Truth based on position
            ground_truth = l1 if frag_idx < split_idx else l2

            for layer_str, preds in entry['predictions'].items():
                layer = int(layer_str)
                if layer not in LAYERS:
                    continue
                
                # Extract probabilities for specific roles
                p_l1 = preds.get(l1, 0.0)
                p_l2 = preds.get(l2, 0.0)
                
                # Compute average of others
                other_sum = 0.0
                other_count = 0
                for label, val in preds.items():
                    if label != l1 and label != l2:
                        other_sum += val
                        other_count += 1
                p_other = other_sum / other_count if other_count > 0 else 0.0
                
                # Compute Accuracy
                if preds:
                    top_label = max(preds, key=preds.get)
                    is_correct = 1.0 if top_label == ground_truth else 0.0
                else:
                    is_correct = 0.0

                # Add to all relevant groups
                for g_name in relevant_groups:
                    grouped_data[g_name]['prob_l1'][layer][frag_idx].append(p_l1)
                    grouped_data[g_name]['prob_l2'][layer][frag_idx].append(p_l2)
                    grouped_data[g_name]['prob_other'][layer][frag_idx].append(p_other)
                    grouped_data[g_name]['acc'][layer][frag_idx].append(is_correct)
    
    print(f"Successfully processed {valid_conv_count} conversations.")

    # --- Comparison Plotting ---
    x_axis = np.arange(1, NUM_FRAGMENTS + 1)
    target_layer = 40
    print(f"Generating comparison plots for Layer {target_layer}...")

    os.makedirs("graphs/change", exist_ok=True)

    COMPARISONS = [
        ("Explicit Switch (Prompts 1,2)", "Implicit Switch (Prompts 3,4)", "comparison_explicit_implicit"),
        ("Pickup (Prompts 1,3)", "No Pickup (Prompts 2,4)", "comparison_pickup_nopickup")
    ]

    for g1_name, g2_name, filename_suffix in COMPARISONS:
        if g1_name not in grouped_data or g2_name not in grouped_data:
            print(f"Skipping comparison {g1_name} vs {g2_name} (data missing)")
            continue

        print(f"Generating comparison: {g1_name} vs {g2_name}...")
        
        # 1. Probe Output Value Plot
        output_plot_val = f"graphs/change/change_probe_dynamics_{category}_{filename_suffix}{suffix}_value.png"
        
        plt.figure(figsize=(12, 7))
        
        # Plot Group 1 (Solid)
        means_l1_g1 = [np.mean(grouped_data[g1_name]['prob_l1'][target_layer][f]) if grouped_data[g1_name]['prob_l1'][target_layer][f] else 0.0 for f in range(NUM_FRAGMENTS)]
        means_l2_g1 = [np.mean(grouped_data[g1_name]['prob_l2'][target_layer][f]) if grouped_data[g1_name]['prob_l2'][target_layer][f] else 0.0 for f in range(NUM_FRAGMENTS)]
        
        plt.plot(x_axis, means_l1_g1, marker='o', color='green', linestyle='-', label=f"{g1_name} - L1", linewidth=2)
        plt.plot(x_axis, means_l2_g1, marker='o', color='blue', linestyle='-', label=f"{g1_name} - L2", linewidth=2)
        
        # Plot Group 2 (Dashed)
        means_l1_g2 = [np.mean(grouped_data[g2_name]['prob_l1'][target_layer][f]) if grouped_data[g2_name]['prob_l1'][target_layer][f] else 0.0 for f in range(NUM_FRAGMENTS)]
        means_l2_g2 = [np.mean(grouped_data[g2_name]['prob_l2'][target_layer][f]) if grouped_data[g2_name]['prob_l2'][target_layer][f] else 0.0 for f in range(NUM_FRAGMENTS)]
        
        plt.plot(x_axis, means_l1_g2, marker='s', color='green', linestyle='--', label=f"{g2_name} - L1", linewidth=2, alpha=0.7)
        plt.plot(x_axis, means_l2_g2, marker='s', color='blue', linestyle='--', label=f"{g2_name} - L2", linewidth=2, alpha=0.7)

        plt.axhline(y=random_chance, color='r', linestyle=':', label=f'Random Chance ({random_chance:.2f})')
        plt.axvline(x=3.5, color='black', linestyle='-', alpha=0.3, label="Switch Point")
        
        # Shading
        plt.axvspan(0.5, 3.5, color='green', alpha=0.05)
        plt.axvspan(3.5, 6.5, color='blue', alpha=0.05)
        
        plt.title(f"{category.capitalize()} - Comparison - Probe Output Value{suffix}")
        plt.xlabel("Turn Number")
        plt.ylabel("Probe Output Value")
        plt.ylim(0, 1.0)
        plt.xticks(x_axis)
        plt.xlim(0.5, 6.5)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(output_plot_val)
        plt.close()
        
        # 2. Accuracy Plot (Comparison)
        output_plot_acc = f"graphs/change/change_probe_dynamics_{category}_{filename_suffix}{suffix}_accuracy.png"
        
        plt.figure(figsize=(12, 7))
        
        means_acc_g1 = [np.mean(grouped_data[g1_name]['acc'][target_layer][f]) if grouped_data[g1_name]['acc'][target_layer][f] else 0.0 for f in range(NUM_FRAGMENTS)]
        means_acc_g2 = [np.mean(grouped_data[g2_name]['acc'][target_layer][f]) if grouped_data[g2_name]['acc'][target_layer][f] else 0.0 for f in range(NUM_FRAGMENTS)]
        
        plt.plot(x_axis, means_acc_g1, marker='o', color='black', linestyle='-', label=f"{g1_name} Accuracy", linewidth=2)
        plt.plot(x_axis, means_acc_g2, marker='s', color='gray', linestyle='--', label=f"{g2_name} Accuracy", linewidth=2)
        
        plt.axhline(y=random_chance, color='r', linestyle=':', label=f'Random Chance')
        plt.axvline(x=3.5, color='black', linestyle='-', alpha=0.3)
        
        plt.axvspan(0.5, 3.5, color='green', alpha=0.05)
        plt.axvspan(3.5, 6.5, color='blue', alpha=0.05)

        plt.title(f"{category.capitalize()} - Comparison - Top-1 Accuracy{suffix}")
        plt.xlabel("Turn Number")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.0)
        plt.xticks(x_axis)
        plt.xlim(0.5, 6.5)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(output_plot_acc)
        plt.close()
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze probe dynamics for attribute change datasets.")
    parser.add_argument("--category", type=str, default="age", choices=list(CATEGORY_CONFIG.keys()), 
                        help="Category to analyze (e.g., age, gender, socioeconomic, education)")
    parser.add_argument("--current_probe", action="store_true", help="If set, uses files with '_current' suffix")
    
    args = parser.parse_args()
    analyze_change_dynamics(args.category, current_probe=args.current_probe)
