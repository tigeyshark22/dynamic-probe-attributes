
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from probes import LinearProbeClassification
# Import utils from dataset.py
from dataset import split_conversation, llama_v2_prompt
import numpy as np
import glob
import json
from tqdm import tqdm
import re

def load_model_and_tokenizer(model_name="meta-llama/Llama-2-13b-chat-hf"):
    print(f"Loading model: {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right' # Important for batching with offset mapping logic below
    
    model = AutoModelForCausalLM.from_pretrained(model_name, token=True)
    # Priority: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"Using device: {device}", flush=True)
    model.half().to(device)
    model.eval()
    return model, tokenizer, device

def standardize_text_dataset_utils(text):
    """
    Uses dataset.py utils (split_conversation, llama_v2_prompt) to standardize text.
    """
    # 1. Split conversation
    # Try standard headers found in checking
    user_msgs, ai_msgs = [], []
    
    if "### Human:" in text:
        user_msgs, ai_msgs = split_conversation(text, "### Human:", "### Assistant:")
    elif "### User:" in text:
        user_msgs, ai_msgs = split_conversation(text, "### User:", "### Assistant:")
    elif "HUMAN:" in text:
        user_msgs, ai_msgs = split_conversation(text, "HUMAN:", "ASSISTANT:")
    else:
        # Default fallback or single message
        # If no headers found, treat whole text as one user message?
        user_msgs = [text]
        ai_msgs = []
        
    # 2. Construct messages list for llama_v2_prompt
    messages_dict = []
    # zip stops at shortest, so if user has 1 more msg, it might be cut if we strictly zip
    # usually conversation starts with User.
    
    # Handle case where lengths differ (usually User is 1 ahead or equal)
    # split_conversation returns lists.
    
    # We need to interleave them
    # Assuming User starts.
    for i in range(max(len(user_msgs), len(ai_msgs))):
        if i < len(user_msgs):
            messages_dict.append({'role': 'user', 'content': user_msgs[i]})
        if i < len(ai_msgs):
            messages_dict.append({'role': 'assistant', 'content': ai_msgs[i]})
            
    # 3. Apply Llama 2 prompt formatting
    # This adds system prompt and [INST] tags
    try:
        formatted_text = llama_v2_prompt(messages_dict)
    except IndexError:
        # llama_v2_prompt might fail if empty or weird structure
        if not messages_dict:
             return text # Return raw if fail
        formatted_text = text # Fallback
        
    return formatted_text


# Configuration
PROBE_DIR = "../data/probe_checkpoints/reading_probe"
DATASET_DIR = "../data/dataset"
TARGET_LAYERS = list(range(1, 41))

CATEGORY_CONFIG = {
    "age": {"probe_prefix": "age", "label_map": {"child": 0, "adolescent": 1, "adult": 2, "older adult": 3}},
    "gender": {"probe_prefix": "gender", "label_map": {"male": 0, "female": 1}},
    "socioeconomic": {"probe_prefix": "socioeco", "label_map": {"low": 0, "middle": 1, "high": 2}},
    "education": {"probe_prefix": "education", "label_map": {"someschool": 0, "highschool": 1, "collegemore": 2}},
    "emotion": {"probe_prefix": "emotion", "label_map": {"sad": 0, "neutral emotion": 1, "happy": 2}},
    "urgency": {"probe_prefix": "urgency", "label_map": {"panic": 0, "normal urgency": 1, "leisure": 2}}
}

PROMPT_TRANSLATOR = {
    "age": "age",
    "gender": "gender",
    "socioeconomic": "socioeconomic status",
    "education": "education level",
    "emotion": "emotion",
    "urgency": "urgency"
}

# === SELECT CATEGORY HERE ===
SELECTED_CATEGORY = "age"

def load_probes(device, probe_dir, category, layers=TARGET_LAYERS, input_dim=5120):
    config = CATEGORY_CONFIG.get(category)
    if not config:
        raise ValueError(f"Unknown category: {category}")
    
    prefix = config["probe_prefix"]
    num_classes = len(config["label_map"])
    
    probes = {}
    print(f"Loading {category} probes from {probe_dir}...", flush=True)
    for layer in layers:
        probe = LinearProbeClassification(
            device=device,
            probe_class=num_classes,
            input_dim=input_dim,
            logistic=True 
        )
        # Try standard patterns
        checkpoint_path = os.path.join(probe_dir, f"{prefix}_probe_at_layer_{layer}.pth")
        if not os.path.exists(checkpoint_path):
             checkpoint_path = os.path.join(probe_dir, f"{prefix}_probe_at_layer_{layer}_final.pth")
        
        if os.path.exists(checkpoint_path):
            try:
                probe.load_state_dict(torch.load(checkpoint_path, map_location=device))
                probe.eval()
                probes[layer] = probe
            except Exception as e:
                print(f"Error loading probe layer {layer}: {e}")
        else:
            print(f"Warning: Probe checkpoint not found for layer {layer} at {checkpoint_path}")
    return probes, config["label_map"]


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            self.out = outputs[0].detach()
        else:
            self.out = outputs.detach()

def run_batched_inference(model, tokenizer, device, category=SELECTED_CATEGORY, batch_size=1, add_prediction_suffix=True, dataset_dir=DATASET_DIR, use_current_suffix=False, use_current_probes=False):
    # 1. Load Probes
    # Determine directory
    if use_current_probes:
        probe_dir = "../data/probe_checkpoints/current_reading_probe"
        # Enforce current suffix if current probes are used, unless explicitly overriden? 
        # Usually they go together. Let's respect the flag if passed, but default to True if use_current_probes is True?
        # User requested modification: "modify run age probes to run with the current reading probes".
        # We can assume if use_current_probes is set, we prefer the 'current' logic.
        if not use_current_suffix: 
             use_current_suffix = True # Auto-enable suffix logic for current probes
    else:
        probe_dir = PROBE_DIR

    probes, label_map = load_probes(device, probe_dir, category)
    id_to_label = {v: k for k, v in label_map.items()}
    
    # 2. Find Files
    # Support both old flat structure and new recursive structure
    # Search for any .txt file under the category directory (or matching *category*)
    
    # Try strict category folder first (e.g. .../age/...)
    category_path = os.path.join(dataset_dir, category)
    if os.path.exists(category_path):
        search_pattern = os.path.join(category_path, "**", "*.txt")
    else:
        # Fallback to wildcard search (e.g. .../*age*/**/*.txt)
        search_pattern = os.path.join(dataset_dir, f"*{category}*", "**", "*.txt")
        
    print(f"Searching for files with pattern: {search_pattern}")
    all_files = glob.glob(search_pattern, recursive=True)
    
    # Filter out empty or temp files if necessary, but removing "openai" filter as requested
    files = [f for f in all_files if os.path.isfile(f) and f.endswith(".txt")]
    
    print(f"Found {len(files)} conversation files for category '{category}' in '{dataset_dir}'")
    
    if not files:
        print("No files found. Exiting.")
        return
        
    # 3. Register Hooks
    # We need to capture outputs of the specific layers.
    # Llama-2 structure: model.model.layers[i]
    hooks = {}
    handles = []
    
    # Identify which layers we need from the loaded probes
    needed_layers = list(probes.keys())
    print(f"Attaching hooks to layers: {needed_layers}")
    
    for layer_idx in needed_layers:
        hook = Hook()
        hooks[layer_idx] = hook
        # Register forward hook on the specific decoder layer
        # Probe index matches hidden_states index.
        # hidden_states[0] = embeddings
        # hidden_states[k] = output of layer k-1
        # So for probe layer_idx, we want output of model layer layer_idx-1
        model_layer_idx = layer_idx - 1
        
        if 0 <= model_layer_idx < len(model.model.layers):
            handle = model.model.layers[model_layer_idx].register_forward_hook(hook)
            handles.append(handle)
        else:
             print(f"Warning: Skipping hook for probe {layer_idx}, mapped to invalid model layer {model_layer_idx}")
        
    # 4. Processing
    results = []
    # Output file name based on dataset dir to avoid overwriting?
    # Or just probe_results_{category}_change.jsonl if dataset_dir suggests it?
    # For now keep same output or maybe append a suffix if it's the change dataset?
    # Let's keep it simple for now, but maybe warn user.
    
    suffix_label = "_current" if use_current_suffix else ""
    
    if "change_dataset" in dataset_dir:
         output_file = f"../data/probe_results_{category}_change{suffix_label}.jsonl"
    else:
         output_file = f"../data/probe_results_{category}{suffix_label}.jsonl"
    
    print("Preparing fragments...")
    all_fragments_text = []
    all_fragments_meta = [] # (fpath, fragment_index)
    
    # 4a. Read and Fragment ALL files
    # Loading all strings into memory is generally fine for text datasets of this size (<1GB text).
    for fpath in tqdm(files, desc="Loading Files"):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                std_text = standardize_text_dataset_utils(raw_text)
                
                # Find all Assistant turn ends (</s>)
                eos_token = "</s>"
                turn_end_indices = [m.end() for m in re.finditer(re.escape(eos_token), std_text)]
                
                if not turn_end_indices:
                    turn_end_indices = [len(std_text)]
                    
                for frag_i, end_idx in enumerate(turn_end_indices):
                    fragment_text = std_text[:end_idx]
                    
                    if add_prediction_suffix:
                            if use_current_suffix:
                                suffix = f" I think the current {PROMPT_TRANSLATOR[category]} of this user is"
                            else:
                                suffix = f" I think the {PROMPT_TRANSLATOR[category]} of this user is"
                            fragment_text += suffix
                            
                    all_fragments_text.append(fragment_text)
                    all_fragments_meta.append((fpath, frag_i))
        except Exception as e:
            print(f"Error reading file {fpath}: {e}")

    print(f"Total fragments generated: {len(all_fragments_text)}")
    print(f"Starting inference with batch size {batch_size}...")

    try:
        # 4b. Run Inference in batches of size `batch_size` (fragments)
        for i in tqdm(range(0, len(all_fragments_text), batch_size), desc="Inference"):
            batch_text_chunk = all_fragments_text[i:i+batch_size]
            batch_meta_chunk = all_fragments_meta[i:i+batch_size]
            
            # Run Model
            try:
                inputs = tokenizer(batch_text_chunk, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            except Exception as e:
                print(f"Error tokenizing batch starting at {i}: {e}")
                continue

            # Clear previous hook outputs
            for h in hooks.values():
                h.out = None

            # Forward pass
            try:
                with torch.no_grad():
                    model(**inputs, output_hidden_states=False)
            except RuntimeError as e:
                 if "out of memory" in str(e).lower():
                     print(f"OOM at batch starting {i}. Skipping batch.")
                     torch.cuda.empty_cache()
                     continue
                 else:
                     raise e

            # Extract Probes
            seq_lens = inputs.attention_mask.sum(dim=1) 
            
            for b_i, (fpath, frag_i) in enumerate(batch_meta_chunk):
                last_token_idx = seq_lens[b_i] - 1
                
                # Construct a unique logical filename that includes metadata from the path
                # This is crucial for datasets like 'change_dataset' where duplicate filenames exist in different folders
                try:
                    rel_path = os.path.relpath(fpath, dataset_dir)
                    # Replace path separators with underscores to make it a flat identifier
                    unique_filename = rel_path.replace(os.sep, "_")
                except ValueError:
                    # Fallback if fpath is not relative to dataset_dir
                    unique_filename = os.path.basename(fpath)

                entry = {
                    "file": unique_filename,
                    "path": fpath,
                    "fragment_index": frag_i,
                    "predictions": {}
                }
                
                for layer, probe in probes.items():
                    # Retrieve hidden state from hook
                    layer_hidden = hooks[layer].out
                    
                    # hidden: [batch, seq_len, hidden_dim]
                    hidden = layer_hidden[b_i, last_token_idx, :].unsqueeze(0).float()
                    
                    with torch.no_grad():
                        logits, _ = probe(hidden)
                        # No softmax, as probe (logistic=True) output is already Sigmoid probabilities.
                        # We just take the raw probabilities.
                        probs = logits.detach().cpu().numpy()[0]
                        # probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
                        entry["predictions"][layer] = {id_to_label[k]: float(p) for k, p in enumerate(probs)}
                
                results.append(entry)
                
    finally:
        # Cleanup handles
        for h in handles:
            h.remove()
                
    # Write Results
    print(f"Writing results to {output_file}...")
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run batched inference with probes.")
    parser.add_argument("--category", type=str, default="age", choices=["age", "gender", "socioeconomic", "education", "emotion", "urgency"], help="Category of probes to run.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--add_prediction_suffix", action="store_true", default=True, help="Append ' I think the {attribute} of this user is' to text. Default: True")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR, help="Path to the dataset directory.")
    parser.add_argument("--use_current_suffix", action="store_true", help="If set, uses 'I think the *current* {attribute}...' as suffix.")
    parser.add_argument("--use_current_probes", action="store_true", help="Use 'current_reading_probe' checkpoints and enforce current suffix.")
    
    args = parser.parse_args()
    
    # Auto-adjust dataset for emotion/urgency if still default
    if args.category in ["emotion", "urgency"] and args.dataset_dir == DATASET_DIR:
        print(f"Switching dataset_dir to '../data/static_dataset' for category '{args.category}'")
        args.dataset_dir = "../data/static_dataset"

    # Load Model and Tokenizer (Run this once)
    model, tokenizer, device = load_model_and_tokenizer()
    
    # Execute
    run_batched_inference(model, tokenizer, device, category=args.category, 
                          batch_size=args.batch_size, 
                          add_prediction_suffix=args.add_prediction_suffix, 
                          dataset_dir=args.dataset_dir,
                          use_current_suffix=args.use_current_suffix,
                          use_current_probes=args.use_current_probes)
