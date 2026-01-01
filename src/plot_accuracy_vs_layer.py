
import matplotlib.pyplot as plt
import os
import numpy as np

INPUT_FILE = "accuracy_per_turn.txt"
OUTPUT_FILE = "graphs/accuracy_vs_layer_per_turn.png"

def parse_data(filepath):
    layers = []
    # Dictionary to hold lists for each turn (0-5)
    turn_data = {i: {"mean": [], "ci": []} for i in range(6)}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    start_parsing = False
    for line in lines:
        if "Layer" in line and "Mean" in line:
            start_parsing = True
            continue
        if "---" in line:
            continue
            
        if start_parsing:
            parts = [p.strip() for p in line.strip().split('|')]
            if len(parts) >= 13: # Layer + 6*(Mean+CI)
                try:
                    l = int(parts[0])
                    layers.append(l)
                    
                    for i in range(6):
                        # Indices: 1, 2 for T1; 3, 4 for T2; etc.
                        idx_mean = 1 + (i * 2)
                        idx_ci = 2 + (i * 2)
                        
                        m = float(parts[idx_mean])
                        c = float(parts[idx_ci])
                        
                        turn_data[i]["mean"].append(m)
                        turn_data[i]["ci"].append(c)
                except ValueError:
                    continue
                    
    return layers, turn_data

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    layers, turn_data = parse_data(INPUT_FILE)
    
    if not layers:
        print("No data parsed.")
        return

    layers = np.array(layers)
    
    # Use a colormap for distinct but related colors
    colors = plt.cm.viridis(np.linspace(0, 1, 6))

    plt.figure(figsize=(12, 7))
    
    for i in range(6):
        mean_vals = np.array(turn_data[i]["mean"])
        ci_vals = np.array(turn_data[i]["ci"])
        
        label_text = f"Turn {i+1}"
        if i == 3: label_text += " (Persona Change)"
        
        plt.plot(layers, mean_vals, label=label_text, color=colors[i], marker='.', linewidth=2)
        plt.fill_between(layers, mean_vals - ci_vals, mean_vals + ci_vals, color=colors[i], alpha=0.15)
    
    plt.xlabel('Layer')
    plt.ylabel('Average Accuracy')
    plt.title('Probe Accuracy vs Layer per Turn (Mean ± 95% Margin of Error)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    plt.savefig(OUTPUT_FILE)
    print(f"Plot saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
