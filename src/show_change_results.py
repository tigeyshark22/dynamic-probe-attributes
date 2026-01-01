
import json

target_file = "age_prompt_3_adult_to_adolescent_conversation_9.txt"
input_file = "../data/probe_results_age_change.jsonl"

print(f"Loading results for {target_file} from {input_file}...\n")

entries = []
with open(input_file, 'r') as f:
    for line in f:
        try:
            entry = json.loads(line)
            if entry['file'] == target_file:
                entries.append(entry)
        except:
            continue

entries.sort(key=lambda x: x['fragment_index'])

print(f"{'Frag':<4} | {'Layer':<5} | {'Child':<7} | {'Adoles':<7} | {'Adult':<7} | {'Older':<7}")
print("-" * 55)

for entry in entries:
    frag_i = entry['fragment_index']
    # Show layers 36-40
    for layer in range(36, 41):
        l_str = str(layer)
        preds = entry['predictions'].get(l_str, {})
        
        p_child = preds.get('child', 0)
        p_adol = preds.get('adolescent', 0)
        p_adult = preds.get('adult', 0)
        p_older = preds.get('older adult', 0)
        
        # Highlight max
        # A simple star or just the numbers
        scores = [p_child, p_adol, p_adult, p_older]
        max_score = max(scores)
        
        def fmt(val):
            s = f"{val:.3f}"
            if val == max_score:
                return f"*{s}*"
            return f" {s} "

        print(f"{frag_i:<4} | {layer:<5} | {fmt(p_child)} | {fmt(p_adol)} | {fmt(p_adult)} | {fmt(p_older)}")
    print("-" * 55)
