
import json
import re
import os
import argparse

def split_conversation(text, user_identifier="HUMAN:", ai_identifier="ASSISTANT:"):
    user_messages = []
    assistant_messages = []

    lines = text.split("\n")

    current_user_message = ""
    current_assistant_message = ""

    for line in lines:
        line = line.lstrip(" ")
        if line.startswith(user_identifier):
            if current_assistant_message:
                assistant_messages.append(current_assistant_message.strip())
                current_assistant_message = ""
            current_user_message += line.replace(user_identifier, "").strip() + " "
        elif line.startswith(ai_identifier):
            if current_user_message:
                user_messages.append(current_user_message.strip())
                current_user_message = ""
            current_assistant_message += line.replace(ai_identifier, "").strip() + " "

    if current_user_message:
        user_messages.append(current_user_message.strip())
    if current_assistant_message:
        assistant_messages.append(current_assistant_message.strip())

    return user_messages, assistant_messages

def llama_v2_prompt(messages):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    # Ensure system prompt is there
    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
        
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    if messages[-1]["role"] == "user":
        messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)

def standardize_text(text):
    user_msgs, ai_msgs = [], []
    
    if "### Human:" in text:
        user_msgs, ai_msgs = split_conversation(text, "### Human:", "### Assistant:")
    elif "### User:" in text:
        user_msgs, ai_msgs = split_conversation(text, "### User:", "### Assistant:")
    elif "HUMAN:" in text:
        user_msgs, ai_msgs = split_conversation(text, "HUMAN:", "ASSISTANT:")
    else:
        user_msgs = [text]
        ai_msgs = []
        
    messages_dict = []
    for i in range(max(len(user_msgs), len(ai_msgs))):
        if i < len(user_msgs):
            messages_dict.append({'role': 'user', 'content': user_msgs[i]})
        if i < len(ai_msgs):
            messages_dict.append({'role': 'assistant', 'content': ai_msgs[i]})
            
    try:
        if not messages_dict:
             return text
        formatted_text = llama_v2_prompt(messages_dict)
    except Exception as e:
        print(f"Error formatting: {e}")
        return text
        
    return formatted_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="gender")
    args = parser.parse_args()
    
    jsonl_path = f"../data/probe_results_{args.category}.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return

    # Load Data
    file_entries = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            fname = entry["file"]
            if fname not in file_entries:
                file_entries[fname] = []
            file_entries[fname].append(entry)

    # Pick a file with > 1 fragments (to show dynamics)
    target_file = None
    # Prioritize files that look like change conversations if possible?
    # The filenames are like 'conversation_184_gender_male.txt'.
    # Change dataset files might be in a different path or named differently?
    # The user generated change conversations in ../data/change_dataset/..., 
    # but run_age_probes searches in ../data/dataset/*category*/*/*.txt.
    # It filters for "openai".
    # Let's just pick one available file.
    
    for fname, entries in file_entries.items():
        if len(entries) > 2:
            target_file = fname
            break
            
    if not target_file:
         # Fallback to any file
         target_file = list(file_entries.keys())[0]

    entries = sorted(file_entries[target_file], key=lambda x: x["fragment_index"])
    # The path in JSON is relative to where it was run, typically ../data/...
    # If we are in src, ../data/... works relative to src.
    fpath = entries[0]["path"] 
    
    print(f"Visualization for file: {target_file}")
    print(f"Path: {fpath}")
    print("=" * 60)
    
    if not os.path.exists(fpath):
        print(f"Cannot find actual text file at {fpath}")
        # Try to fix path if needed?
        return

    with open(fpath, "r") as f:
        raw_text = f.read()

    formatted_text = standardize_text(raw_text)
    
    # Extract User inputs
    # Regex to find content inside [INST] ... [/INST]
    # This matches the structure created by llama_v2_prompt
    inst_blocks = list(re.finditer(r"\[INST\](.*?)\[/INST\]", formatted_text, re.DOTALL))
    
    for i, entry in enumerate(entries):
        frag_idx = entry["fragment_index"]
        
        print(f"Turn {i+1} (Fragment {frag_idx})")
        
        # prediction at layer 40 (last layer)
        if "40" in entry["predictions"]:
            scores = entry["predictions"]["40"]
            # Format scores
            score_str = ", ".join([f"{k}: {v:.3f}" for k, v in scores.items()])
            print(f"  Probe Scores (Layer 40): {score_str}")
        else:
             print("  Probe Scores: N/A")
             
        if frag_idx < len(inst_blocks):
            content = inst_blocks[frag_idx].group(1).strip()
            # If content includes system prompt at start, maybe hide it?
            # llama_v2_prompt injects system prompt into the first [INST] block.
            # "<<SYS>>...<</SYS>>"
            
            # Clean system prompt for display
            display_content = re.sub(r"<<SYS>>.*?<</SYS>>", "[System Prompt Hidden]", content, flags=re.DOTALL).strip()
            
            print(f"  User Input: \"{display_content}\"")
        else:
            print("  [Could not extract text fragment]")
            
        print("-" * 60)

if __name__ == "__main__":
    main()
