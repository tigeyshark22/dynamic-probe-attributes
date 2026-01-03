
import json
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
DATA_DIR = "../data"
TARGET_LAYER = "40"
IDX_4 = 3
IDX_5 = 4
IDX_6 = 5

PROMPT_TYPES = {
    1: "Explicit - Pickup",
    2: "Explicit - No Pickup",
    3: "Implicit - Pickup",
    4: "Implicit - No Pickup"
}

def get_prompt_type(filename):
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

def is_correct(predictions, target_layer, label):
    preds = predictions.get(target_layer)
    if preds:
        top = max(preds, key=preds.get)
        return top == label
    return False

def calc_moe(p, n):
    if n == 0: return 0.0
    return 1.96 * np.sqrt((p * (1 - p)) / n)

def main():
    files = glob.glob(os.path.join(DATA_DIR, "probe_results_*_change_current.jsonl"))
    print(f"Found {len(files)} result files.")
    
    # Structure: type -> stats dict
    type_stats = {name: {'total': 0, 'inc_4': 0, 'inc_5': 0, 'inc_6': 0, 'inc_4_and_5': 0, 'inc_5_and_6': 0} 
                  for name in PROMPT_TYPES.values()}
    
    for fpath in files:
        # Load file
        file_entries = defaultdict(list)
        with open(fpath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    file_entries[entry['file']].append(entry)
                except json.JSONDecodeError:
                    continue
                    
        for filename, entries in file_entries.items():
            ptype = get_prompt_type(filename)
            if not ptype: continue
            
            _, l1, l2 = get_change_labels(filename)
            if not l2: continue
            
            e4 = next((e for e in entries if e['fragment_index'] == IDX_4), None)
            e5 = next((e for e in entries if e['fragment_index'] == IDX_5), None)
            e6 = next((e for e in entries if e['fragment_index'] == IDX_6), None)
            
            if not (e4 and e5 and e6): continue
            
            stats = type_stats[ptype]
            stats['total'] += 1
            
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
            
    # Calculate Probabilities & Plot
    
    # 1. Marginal Error Rates
    # Data structure for plotting: groups (Types), subgroups (Turns)
    # types = list(type_stats.keys())
    # Fixed order for consistency: Explicit No Pickup, Implicit No Pickup
    ordered_types = ["Explicit - No Pickup", "Implicit - No Pickup"]
    
    # Prepare data for Plot 1 (Marginals)
    # X axis: Turn 4, Turn 5, Turn 6
    # Groups: Types
    turns = ['Turn 4', 'Turn 5', 'Turn 6']
    x = np.arange(len(turns))
    width = 0.35 # Wider bars for fewer groups
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Colors for types (Orange, Red to match previous convention for No Pickup)
    colors = ['#ff7f0e', '#d62728']
    
    print("\nMarginal Error Rates:")
    for i, ptype in enumerate(ordered_types):
        s = type_stats[ptype]
        total = s['total']
        if total == 0: continue
        
        p4 = s['inc_4'] / total
        p5 = s['inc_5'] / total
        p6 = s['inc_6'] / total
        
        err4 = calc_moe(p4, total)
        err5 = calc_moe(p5, total)
        err6 = calc_moe(p6, total)
        
        probs = [p4, p5, p6]
        errs = [err4, err5, err6]
        
        print(f"{ptype}: N={total}, P(Inc)=[4:{p4:.3f}, 5:{p5:.3f}, 6:{p6:.3f}]")
        
        # Plot bars
        offset = (i - 0.5) * width 
        ax1.bar(x + offset, probs, width, yerr=errs, label=ptype, capsize=5, color=colors[i], alpha=0.8)

    ax1.set_ylabel('Probability of Incorrect Prediction')
    ax1.set_title('Marginal Error Rates (No Pickup Only)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(turns)
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Prepare data for Plot 2 (Conditionals)
    # X axis: T4->T5, T5->T6
    transitions = ['T4 Inc → T5 Inc', 'T5 Inc → T6 Inc']
    x2 = np.arange(len(transitions))
    
    print("\nConditional Persistence:")
    for i, ptype in enumerate(ordered_types):
        s = type_stats[ptype]
        
        # P(5|4)
        p5_4 = s['inc_4_and_5'] / s['inc_4'] if s['inc_4'] > 0 else 0
        n5_4 = s['inc_4']
        err5_4 = calc_moe(p5_4, n5_4)
        
        # P(6|5)
        p6_5 = s['inc_5_and_6'] / s['inc_5'] if s['inc_5'] > 0 else 0
        n6_5 = s['inc_5']
        err6_5 = calc_moe(p6_5, n6_5)
        
        probs = [p5_4, p6_5]
        errs = [err5_4, err6_5]
        
        print(f"{ptype}: P(5|4)={p5_4:.3f} (N={n5_4}), P(6|5)={p6_5:.3f} (N={n6_5})")
        
        offset = (i - 0.5) * width
        ax2.bar(x2 + offset, probs, width, yerr=errs, label=ptype, capsize=5, color=colors[i], alpha=0.8)
        
    ax2.set_ylabel('Conditional Probability (Persistence)')
    ax2.set_title('Error Persistence (No Pickup Only)')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(transitions)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = "graphs/layer_40_error_dynamics_no_pickup.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    main()
