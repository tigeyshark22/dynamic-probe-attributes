
import os
import re
import json
import itertools
from tqdm import tqdm
from groq import Groq

# Configuration
PROMPTS_FILE = "../data/mid_conversation_change_prompts_v2.txt"
OUTPUT_DIR = "../data/change_dataset"
MODEL_NAME = "llama-3.3-70b-versatile"

# Category Definitions (Hardcoded based on prompt text)
CATEGORIES = {
    "Gender": {
        "labels": ["male", "female"],
        "key_map": {"label": "gender"},
        "samples": 0 # 2 pairs * 4 prompts * 60 = 480
    },
    "Age": {
        "labels": ["child", "adolescent", "adult", "older adult"],
        "key_map": {"label": "age", "extra": "year_range"},
        "extra_map": {
            "child": "below 12 years old",
            "adolescent": "between 13 to 17 years old",
            "adult": "between 18 to 64 years old",
            "older adult": "above 65 years old"
        },
        "samples": 40 # 12 pairs * 4 prompts * 10 = 480
    },
    "Education": {
        "labels": ["some schooling (elementary school, middle school, or pre-high school)", 
                   "high school education", 
                   "college and more"],
        # Short names for folders
        "short_labels": {
            "some schooling (elementary school, middle school, or pre-high school)": "someschool",
            "high school education": "highschool",
            "college and more": "collegemore"
        },
        "key_map": {"label": "education"},
        "samples": 80 # 6 pairs * 4 prompts * 20 = 480
    },
    "Socioeconomic Status": {
        "labels": ["low", "middle", "high"],
        "key_map": {"label": "socioeco"},
        "samples": 80 # 6 pairs * 4 prompts * 20 = 480
    },
    "Emotion": {
        "labels": ["sad", "neutral emotion", "happy"],
        "key_map": {"label": "emotion"},
        "samples": 50 # 6 pairs * 4 prompts * 20 = 480
    },
    "Urgency": {
        "labels": ["panic", "normal urgency", "leisure"],
        "key_map": {"label": "urgency"},
        "samples": 50 # 6 pairs * 4 prompts * 20 = 480
    }
}

def parse_prompts(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    prompts = {}
    
    # Split by Category sections (A.1, A.2, etc.)
    category_pattern = re.compile(r"A\.\d+\s+([^\n]+)")
    matches = list(category_pattern.finditer(content))
    
    for i, match in enumerate(matches):
        category_name = match.group(1).strip()
        start = match.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(content)
        
        section_content = content[start:end].strip()
        
        if category_name == "System Prompt":
            # Extract LLaMa2 prompt
            llama_marker = "For the LLaMa2Chat-13B model, we used the following system prompt"
            if llama_marker in section_content:
                parts = section_content.split(llama_marker)
                if len(parts) > 1:
                    target_part = parts[1].strip()
                    # Regex to capture content inside quotes (supports both smart quotes and standard quotes)
                    quote_match = re.search(r'[“"](.*?)[”"]', target_part, re.DOTALL)
                    if quote_match:
                        prompts["_SYSTEM_PROMPT"] = quote_match.group(1)
            continue
            
        # Parse numbered prompts within section
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
    # Groq API call
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
    3. Handles variations like "Human 2", "User 2" -> "### Human:"
    """
    
    lines = text.split('\n')
    cleaned_lines = []
    
    # Regex to identify headers
    header_regex = re.compile(r'^(?:###\s*)?(Human|User|Assistant)(?:\s+\d+)?\s*:', re.IGNORECASE)
    
    found_first_header = False
    last_role = None
    
    for line in lines:
        line = line.strip()
        match = header_regex.match(line)
        if match:
            role_raw = match.group(1).lower()
            current_normalized_role = 'human' if role_raw in ['human', 'user'] else 'assistant'
            
            # Capture content after the header
            content_after = line[match.end():].strip()
            
            if not found_first_header:
                if current_normalized_role == 'human':
                    found_first_header = True
            
            if found_first_header:
                # Add header only if role changed
                if current_normalized_role != last_role:
                    if current_normalized_role == 'human':
                        cleaned_lines.append("### Human:")
                    else:
                        cleaned_lines.append("### Assistant:")
                    last_role = current_normalized_role
                
                # If there was content on the same line, append it
                if content_after:
                    cleaned_lines.append(content_after)
        else:
            if found_first_header:
                cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate mid-conversation change conversations.")
    parser.add_argument("--prompts_file", type=str, default=PROMPTS_FILE, help="Path to prompts file.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Path to output directory.")
    args = parser.parse_args()

    if not os.path.exists(args.prompts_file):
        print(f"Prompts file not found: {args.prompts_file}")
        return

    # Check for API Key
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set.")
        return

    # Initialize Client
    print("Initializing Groq client...")
    client = Groq()

    parsed_prompts = parse_prompts(args.prompts_file)
    
    system_prompt = parsed_prompts.pop("_SYSTEM_PROMPT", None)
    if system_prompt:
        print(f"Loaded System Prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt found/loaded.")

    print("Parsed categories:", parsed_prompts.keys())
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for category, prompt_dict in parsed_prompts.items():
        if category not in CATEGORIES:
            print(f"Skipping unknown category parsed from text: {category}")
            continue
            
        config = CATEGORIES[category]
        labels = config["labels"]
        short_labels = config.get("short_labels", {})
        samples_per_pair = config.get("samples", 1)
        
        combinations = list(itertools.permutations(labels, 2))
        print(f"Processing Category: {category} ({len(prompt_dict)} prompts, {len(combinations)} label pairs, {samples_per_pair} samples/pair)")
        
        for prompt_id, prompt_template in prompt_dict.items():
            print(f"  Prompt Version {prompt_id}")
            
            for label1, label2 in tqdm(combinations, desc=f"  Generating {category} V{prompt_id}"):
                
                l1_name = short_labels.get(label1, label1).replace(" ", "_")
                l2_name = short_labels.get(label2, label2).replace(" ", "_")
                
                key_map = config["key_map"]
                main_key = key_map["label"] 
                
                replacements = {}
                replacements[f"{main_key}_1"] = label1
                replacements[f"{main_key}_2"] = label2
                
                if "extra" in key_map:
                    extra_key = key_map["extra"]
                    extra_map = config["extra_map"]
                    replacements[f"{extra_key}_1"] = extra_map[label1]
                    replacements[f"{extra_key}_2"] = extra_map[label2]
                
                try:
                    current_prompt = prompt_template.format(**replacements)
                except KeyError as e:
                    print(f"    Error formatting prompt: {e}")
                    continue
                
                comb_name = f"{l1_name}_to_{l2_name}"
                sub_dir = os.path.join(args.output_dir, category.replace(" ", "_").lower(), f"prompt_{prompt_id}", comb_name)
                os.makedirs(sub_dir, exist_ok=True)
                
                # Find next available index
                existing_files = [f for f in os.listdir(sub_dir) if f.startswith("conversation_") and f.endswith(".txt")]
                existing_indices = []
                for f in existing_files:
                    try:
                        # expected format: conversation_X.txt
                        idx = int(f.split("_")[1].split(".")[0])
                        existing_indices.append(idx)
                    except (IndexError, ValueError):
                        continue
                
                next_idx = max(existing_indices) + 1 if existing_indices else 0
                
                # Generate 'samples_per_pair' NEW samples
                print(f"    Appending {samples_per_pair} new conversations starting at index {next_idx}...")
                
                for i in range(next_idx, next_idx + samples_per_pair):
                    fname = f"conversation_{i}.txt"
                    fpath = os.path.join(sub_dir, fname)
                    
                    # Generate
                    raw_text = generate_text(client, current_prompt, system_prompt=system_prompt)
                    
                    if raw_text:
                        # Clean
                        clean_text = clean_and_standardize_conversation(raw_text)
                        
                        with open(fpath, 'w') as f:
                            f.write(clean_text)
                        
    print("Generation complete!")

if __name__ == "__main__":
    main()
