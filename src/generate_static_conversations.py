
import os
import re
import json
from tqdm import tqdm
from groq import Groq

# Configuration
PROMPTS_FILE = "../data/static_emotion_urgency_prompts.txt"
OUTPUT_DIR = "../data/static_dataset"
MODEL_NAME = "llama-3.3-70b-versatile"

# Category Configs
CATEGORIES = {
    "Emotion": {
        "labels": ["sad", "neutral emotion", "happy"],
        "key_map": {"label": "emotion"},
        "samples": 60 # Default samples per label
    },
    "Urgency": {
        "labels": ["panic", "normal urgency", "leisure"],
        "key_map": {"label": "urgency"},
        "samples": 60 # Default samples per label
    }
}

def parse_prompts(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    prompts = {}
    
    # Regex to find sections like "A.1 Emotion"
    category_pattern = re.compile(r"A\.\d+\s+([^\n]+)")
    matches = list(category_pattern.finditer(content))
    
    for i, match in enumerate(matches):
        category_name = match.group(1).strip()
        start = match.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(content)
        
        section_content = content[start:end].strip()
        
        # Parse numbered prompts
        prompt_pattern = re.compile(r"^(\d+)\.\s+(.*?)(?=\n\d+\.|A\.\d+|$)", re.DOTALL | re.MULTILINE)
        
        category_prompts = {}
        for p_match in prompt_pattern.finditer(section_content):
            p_id = p_match.group(1)
            p_text = p_match.group(2).strip()
            category_prompts[int(p_id)] = p_text
            
        if category_prompts:
            prompts[category_name] = category_prompts
            
    return prompts

def generate_text(client, prompt_text, system_prompt=None, max_new_tokens=1024):
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt_text})
        
        completion = client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            max_tokens=max_new_tokens,
            temperature=1.0,
            top_p=0.9,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating text: {e}")
        return ""

def clean_and_standardize_conversation(text):
    """
    Standardizes the conversation format:
    1. Removes introductory text (e.g. "Sure, here is...")
    2. Normalizes headers to "### Human:" and "### Assistant:"
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    header_regex = re.compile(r'^(?:###\s*)?(Human|User|Assistant)(?:\s+\d+)?\s*:', re.IGNORECASE)
    found_first_header = False
    last_role = None
    
    for line in lines:
        line = line.strip()
        match = header_regex.match(line)
        if match:
            role_raw = match.group(1).lower()
            current_normalized_role = 'human' if role_raw in ['human', 'user'] else 'assistant'
            
            content_after = line[match.end():].strip()
            
            if not found_first_header:
                if current_normalized_role == 'human':
                    found_first_header = True
            
            if found_first_header:
                if current_normalized_role != last_role:
                    if current_normalized_role == 'human':
                        cleaned_lines.append("### Human:")
                    else:
                        cleaned_lines.append("### Assistant:")
                    last_role = current_normalized_role
                
                if content_after:
                    cleaned_lines.append(content_after)
        else:
            if found_first_header:
                cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate static conversations.")
    parser.add_argument("--prompts_file", type=str, default=PROMPTS_FILE, help="Path to prompts file.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Path to output directory.")
    parser.add_argument("--samples_per_label", type=int, default=1, help="Number of samples to generate per label.")
    args = parser.parse_args()

    if not os.path.exists(args.prompts_file):
        print(f"Prompts file not found: {args.prompts_file}")
        return

    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set.")
        return

    print("Initializing Groq client...")
    client = Groq()

    parsed_prompts = parse_prompts(args.prompts_file)
    print("Parsed categories:", parsed_prompts.keys())
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for category, prompt_dict in parsed_prompts.items():
        if category not in CATEGORIES:
            print(f"Skipping unknown category parsed from text: {category}")
            continue
            
        config = CATEGORIES[category]
        labels = config["labels"]
        # Allow checking for a 'samples' config, default to 1 (or 60 as defined in CATEGORIES)
        samples_per_label = args.samples_per_label
        
        print(f"Processing Category: {category}")
        
        for prompt_id, prompt_template in prompt_dict.items():
            print(f"  Prompt Version {prompt_id}")
            
            for label in tqdm(labels, desc=f"  Generating {category} V{prompt_id}"):
                
                key_map = config["key_map"]
                main_key = key_map["label"]
                
                replacements = {main_key: label}
                
                try:
                    current_prompt = prompt_template.format(**replacements)
                except KeyError as e:
                    print(f"    Error formatting prompt: {e}")
                    continue
                
                # Output dir: data/static_dataset/{category}/{label}/
                label_dir_name = label.replace(" ", "_")
                sub_dir = os.path.join(args.output_dir, category.replace(" ", "_").lower(), label_dir_name)
                os.makedirs(sub_dir, exist_ok=True)
                
                # Check existance to append
                existing_files = [f for f in os.listdir(sub_dir) if f.startswith("conversation_") and f.endswith(".txt")]
                existing_indices = []
                for f in existing_files:
                    try:
                        idx = int(f.split("_")[1].split(".")[0])
                        existing_indices.append(idx)
                    except (IndexError, ValueError):
                        continue
                
                next_idx = max(existing_indices) + 1 if existing_indices else 0

                print(f"    Appending {samples_per_label} new conversations starting at index {next_idx}...")

                for i in range(next_idx, next_idx + samples_per_label):
                    fname = f"conversation_{i}.txt"
                    fpath = os.path.join(sub_dir, fname)
                    
                    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
                    
                    raw_text = generate_text(client, current_prompt, system_prompt=system_prompt)
                    
                    if raw_text:
                        clean_text = clean_and_standardize_conversation(raw_text)
                        with open(fpath, 'w') as f:
                            f.write(clean_text)
                        
    print("Generation complete!")

if __name__ == "__main__":
    main()
