
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

NUM_BINS = 5
LAYERS = [36, 37, 38, 39, 40]
CATEGORIES = ["age", "gender", "socioeconomic", "education", "emotion", "urgency"]

def get_ground_truth(filename, label_map):
    """
    Robustly extracts the ground truth label from the filename by searching 
    for known labels defined in the category config.
    """
    # Normalize filename: replace underscores with spaces
    base = os.path.basename(filename)
    name_normalized = base.replace("_", " ").replace("-", " ").lower()
    
    # Sort labels by length descending to match "older adult" before "adult"
    sorted_labels = sorted(label_map.keys(), key=len, reverse=True)
    
    for label in sorted_labels:
        # Check if label is in the filename
        # We add spaces around to ensure we don't match substrings incorrectly if possible
        # But simple substring search is usually enough given the distinct labels
        if label in name_normalized:
            return label
            
    return None

def load_category_data(category, current_probe=False):
    config = CATEGORY_CONFIG.get(category)
    if not config:
        return None, None, 0

    suffix = "_current" if current_probe else ""
    input_file = f"../data/probe_results_{category}{suffix}.jsonl"
    prob_means = {} # layer -> list of means
    
    # Use global NUM_BINS
    num_bins = NUM_BINS
        
    if not os.path.exists(input_file):
        print(f"File {input_file} not found. Skipping {category}.")
        return None, config["chance"], num_bins
    
    aggregated_prob = collections.defaultdict(lambda: collections.defaultdict(list))
    conversations = collections.defaultdict(list)

    print(f"Reading {input_file} (Mode: Normalized to {num_bins} bins)...")
    try:
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    conversations[entry['path']].append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return None, config["chance"], num_bins

    total_convs = len(conversations)

    for path, entries in conversations.items():
        if not entries: continue
        entries.sort(key=lambda x: x['fragment_index'])
        
        ground_truth = get_ground_truth(entries[0]['file'], config["label_map"])
        if not ground_truth:
             continue
            
        total_fragments = len(entries)
        if total_fragments == 0: continue

        for i, entry in enumerate(entries):
            # Normalized mapping
            normalized_pos = i / total_fragments
            bin_idx = int(normalized_pos * num_bins)
            if bin_idx >= num_bins: bin_idx = num_bins - 1
            
            for layer_str, preds in entry['predictions'].items():
                layer = int(layer_str)
                if layer not in LAYERS: continue
                
                prob = preds.get(ground_truth, 0.0)
                aggregated_prob[layer][bin_idx].append(prob)
    
    print(f"  Using {total_convs} conversations")

    for layer in LAYERS:
        means = []
        for b in range(num_bins):
            vals = aggregated_prob[layer][b]
            mean_val = np.mean(vals) if vals else 0.0
            means.append(mean_val)
        prob_means[layer] = means
        
    return prob_means, config["chance"], num_bins

def plot_all_grid(current_probe=True):
    suffix = "_current" if current_probe else ""
    
    # Create 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten() # Flatten to 1D array for easy iteration
    
    for i, category in enumerate(CATEGORIES):
        ax = axes[i]
        print(f"Loading data for {category}...")
        
        data, chance, num_bins = load_category_data(category, current_probe)
        
        ax.set_title(f"{category.capitalize()}")
        ax.set_xlabel("Progress (%)")
        ax.set_ylabel("Prob")
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        # Plot Random Chance
        ax.axhline(y=chance, color='r', linestyle=':', label=f'Chance ({chance:.2f})')
        
        if data and num_bins > 0:
            x_axis = np.linspace(0, 100, num_bins)
            for layer in sorted(data.keys()):
                # Highlight Layer 40
                if layer == 40:
                    ax.plot(x_axis, data[layer], marker='o', linewidth=2, label=f"L{layer}")
                else:
                    ax.plot(x_axis, data[layer],  linestyle='--', alpha=0.7, label=f"L{layer}")
        else:
            ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        ax.legend(fontsize='small', loc='lower right')

    plt.tight_layout()
    plt.suptitle(f"Probe Probability Dynamics Across Categories (layers {min(LAYERS)}-{max(LAYERS)}){suffix}", y=1.02, fontsize=16)
    
    output_path = f"graphs/static/probe_dynamics_grid_all{suffix}.png"
    os.makedirs("graphs/static", exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Grid plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2x3 grid of probe dynamics.")
    parser.add_argument("--current_probe", action="store_true", default=True, help="Use current probe data (default: True)")
    parser.add_argument("--no_current_probe", action="store_false", dest="current_probe", help="Use original probe data")
    
    args = parser.parse_args()
    
    plot_all_grid(current_probe=args.current_probe)
