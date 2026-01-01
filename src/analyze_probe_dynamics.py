
import json
import os
import glob
import numpy as np
import collections
import matplotlib.pyplot as plt
import argparse

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

NUM_BINS = 10
LAYERS = [36, 37, 38, 39, 40]

def get_ground_truth(filename, prefix):
    """
    Extracts the ground truth label from the filename.
    Expected format: conversation_{id}_{prefix}_{label}.txt
    """
    base = os.path.basename(filename)
    name_without_ext = os.path.splitext(base)[0]
    parts = name_without_ext.split("_")
    # Strategy: look for the prefix keyword and take the rest
    try:
        # Find all occurrences of prefix, take the last one to be safe if prefix appears earlier
        indices = [i for i, x in enumerate(parts) if x == prefix]
        if not indices:
             return None
        
        prefix_idx = indices[-1] # Use the last occurrence
        label = " ".join(parts[prefix_idx+1:])
        return label
    except ValueError:
        return None

def analyze_dynamics(category, current_probe=False):
    config = CATEGORY_CONFIG.get(category)
    if not config:
        print(f"Error: Unknown category '{category}'")
        return

    suffix = "_current" if current_probe else ""
    input_file = f"../data/probe_results_{category}{suffix}.jsonl"
    os.makedirs("graphs/static", exist_ok=True)
    output_plot_prob = f"graphs/static/probe_dynamics_prob_{category}{suffix}.png"
    output_plot_acc = f"graphs/static/probe_dynamics_acc_{category}{suffix}.png"
    random_chance = config["chance"]
    prefix = config["prefix"]

    print(f"Analyzing Category: {category}")
    print(f"Input File: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return

    # Data structure:
    # layer -> bin_index -> list of probabilities of correct label
    aggregated_prob = collections.defaultdict(lambda: collections.defaultdict(list))
    # layer -> bin_index -> list of 1s (correct) and 0s (incorrect)
    aggregated_acc = collections.defaultdict(lambda: collections.defaultdict(list))
    
    # helper to track conversation lengths to normalize
    # conv_path -> list of entries (to sort by fragment_index)
    conversations = collections.defaultdict(list)

    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                conversations[entry['path']].append(entry)
            except json.JSONDecodeError:
                continue

    print(f"Processing {len(conversations)} conversations...")
    
    for path, entries in conversations.items():
        # Sort by fragment index to ensure order
        entries.sort(key=lambda x: x['fragment_index'])
        
        ground_truth = get_ground_truth(entries[0]['file'], prefix)
        
        # Verify label matches known labels (optional check)
        # if ground_truth not in config["label_map"]: ...
        
        if not ground_truth:
            print(f"Warning: Could not extract label from {entries[0]['file']} with prefix '{prefix}'")
            continue
            
        total_fragments = len(entries)
        if total_fragments == 0:
            continue

        for i, entry in enumerate(entries):
            # Calculate progress (0.0 to 1.0)
            normalized_pos = i / total_fragments
            bin_idx = int(normalized_pos * NUM_BINS)
            if bin_idx >= NUM_BINS:
                bin_idx = NUM_BINS - 1
            
            for layer_str, preds in entry['predictions'].items():
                layer = int(layer_str)
                if layer not in LAYERS:
                    continue
                
                # Get probability of ground truth
                prob = preds.get(ground_truth, 0.0)
                aggregated_prob[layer][bin_idx].append(prob)
                
                # Get accuracy (is top prediction correct?)
                if preds:
                    top_label = max(preds, key=preds.get)
                    is_correct = 1.0 if top_label == ground_truth else 0.0
                else:
                    is_correct = 0.0
                aggregated_acc[layer][bin_idx].append(is_correct)

    # Compute means for Probabilities
    layer_means_prob = {}
    
    print("\nResults (Average Probability of Correct Label):")
    print(f"{'Layer':<6} | " + " | ".join([f"Bin {b}" for b in range(NUM_BINS)]))
    print("-" * (10 + 8 * NUM_BINS))

    for layer in sorted(aggregated_prob.keys()):
        means = []
        for b in range(NUM_BINS):
            vals = aggregated_prob[layer][b]
            mean_val = np.mean(vals) if vals else 0.0
            means.append(mean_val)
        
        layer_means_prob[layer] = means
        print(f"{layer:<6} | " + " | ".join([f"{m:.3f}" for m in means]))

    # Compute means for Accuracy
    layer_means_acc = {}
    
    print("\nResults (Accuracy - Top matching Ground Truth):")
    print(f"{'Layer':<6} | " + " | ".join([f"Bin {b}" for b in range(NUM_BINS)]))
    print("-" * (10 + 8 * NUM_BINS))

    for layer in sorted(aggregated_acc.keys()):
        means = []
        for b in range(NUM_BINS):
            vals = aggregated_acc[layer][b]
            mean_val = np.mean(vals) if vals else 0.0
            means.append(mean_val)
        
        layer_means_acc[layer] = means
        print(f"{layer:<6} | " + " | ".join([f"{m:.3f}" for m in means]))

    x_axis = np.linspace(0, 100, NUM_BINS)

    # Plotting Probabilities
    print(f"\nGenerating plot: {output_plot_prob}...")
    plt.figure(figsize=(10, 6))
    plt.axhline(y=random_chance, color='r', linestyle=':', label=f'Random Chance ({random_chance:.3f})')
    for layer in sorted(layer_means_prob.keys()):
        plt.plot(x_axis, layer_means_prob[layer], marker='o', label=f"Layer {layer}")
    
    plt.title(f"{category.capitalize()} Probe Probability Dynamics over Conversation Progress{suffix}")
    plt.xlabel("Conversation Progress (%)")
    plt.ylabel("Probability of Correct Label")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_plot_prob)
    
    # Plotting Accuracy
    print(f"Generating plot: {output_plot_acc}...")
    plt.figure(figsize=(10, 6))
    plt.axhline(y=random_chance, color='r', linestyle=':', label=f'Random Chance ({random_chance:.3f})')
    for layer in sorted(layer_means_acc.keys()):
        plt.plot(x_axis, layer_means_acc[layer], marker='x', linestyle='--', label=f"Layer {layer}")
    
    plt.title(f"{category.capitalize()} Probe Accuracy Dynamics over Conversation Progress{suffix}")
    plt.xlabel("Conversation Progress (%)")
    plt.ylabel("Accuracy (Top Label == Ground Truth)")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_plot_acc)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze probe dynamics for a specific category.")
    parser.add_argument("--category", type=str, default="age", choices=list(CATEGORY_CONFIG.keys()), 
                        help="Category to analyze (e.g., age, gender, socioeconomic, education)")
    parser.add_argument("--current_probe", action="store_true", help="If set, uses files with '_current' suffix")
    
    args = parser.parse_args()
    analyze_dynamics(args.category, current_probe=args.current_probe)

