
import os
import torch
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset, Subset
from dataset import TextDataset
from probes import LinearProbeClassification, TrainerConfig
from train_test_utils import train, test
import argparse
import sklearn.model_selection
import numpy as np
import torch.nn as nn

# Configuration
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
DATASET_DIR = "../data/dataset"
OUTPUT_DIR = "../data/probe_checkpoints/current_reading_probe"
LAYERS = list(range(1, 41))

CATEGORY_CONFIG = {
    "age": {"probe_prefix": "age", "label_map": {"child": 0, "adolescent": 1, "adult": 2, "older adult": 3}},
    "gender": {"probe_prefix": "gender", "label_map": {"male": 0, "female": 1}},
    "socioeconomic": {"probe_prefix": "socioeco", "label_map": {"low": 0, "middle": 1, "high": 2}},
    "education": {"probe_prefix": "education", "label_map": {"someschool": 0, "highschool": 1, "collegemore": 2}},
    "emotion": {"probe_prefix": "emotion", "label_map": {"sad": 0, "neutral emotion": 1, "happy": 2}}, # Note: "neutral_emotion" -> "neutral emotion" conversion handled in dataset.py
    "urgency": {"probe_prefix": "urgency", "label_map": {"panic": 0, "normal urgency": 1, "leisure": 2}}
}


def load_model_and_tokenizer(model_name=MODEL_NAME):
    print(f"Loading model: {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right' 
    
    model = AutoModelForCausalLM.from_pretrained(model_name, token=True)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"Using device: {device}", flush=True)
    model.half().to(device)
    model.eval()
    return model, tokenizer, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="age", choices=list(CATEGORY_CONFIG.keys()))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    
    config = CATEGORY_CONFIG[args.category]
    
    # Configure defaults for static dataset (emotion/urgency) vs original dataset
    label_from_folder = False
    
    if args.category in ["emotion", "urgency"]:
        # If user hasn't manually overridden the default dataset dir (checked by string equality to constant)
        if args.dataset_dir == DATASET_DIR: 
            args.dataset_dir = "../data/static_dataset"
        label_from_folder = True
    
    # Dynamic Directory Search
    search_pattern = os.path.join(args.dataset_dir, f"*{args.category}*")
    matching_paths = glob.glob(search_pattern)
    category_dirs = [p for p in matching_paths if os.path.isdir(p)]
    
    if not category_dirs:
        # Fallback: maybe the folder is exactly the category name (no wildcards)
        exact_path = os.path.join(args.dataset_dir, args.category)
        if os.path.isdir(exact_path):
             category_dirs = [exact_path]
        else:
             raise FileNotFoundError(f"Could not find any directories matching {search_pattern} or {exact_path}")
        
    print(f"Found {len(category_dirs)} directories for category '{args.category}':")
    for d in category_dirs:
        print(f" - {d}")
        
    primary_dir = category_dirs[0]
    additional_dirs = category_dirs[1:] if len(category_dirs) > 1 else None

    # Load Model (Always, since caching is removed)
    model, tokenizer, device = load_model_and_tokenizer()
    
    print(f"Loading Dataset for {args.category} (with use_current_suffix=True)...")
    dataset = TextDataset(
        directory=primary_dir,
        tokenizer=tokenizer,
        model=model,
        label_idf=f"_{config['probe_prefix']}_", 
        label_to_id=config["label_map"],
        convert_to_llama2_format=True,
        # User requested arguments:
        new_format=True,
        residual_stream=True, 
        if_augmented=False,
        remove_last_ai_response=False,
        include_inst=True,
        one_hot=False,
        # Other settings:
        use_current_suffix=True,
        additional_datas=additional_dirs,
        label_from_folder=label_from_folder
    )
    
    print(f"Dataset Size: {len(dataset)}")
    
    # Stratified Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    labels_list = dataset.labels
    # Convert one-hot to indices for stratification
    stratify_labels = []
    if len(labels_list) > 0:
        sample_label = labels_list[0]
        if isinstance(sample_label, torch.Tensor):
            if sample_label.numel() > 1: # One-hot or multi-dim
                stratify_labels = [torch.argmax(l).item() for l in labels_list]
            else:
                 stratify_labels = [l.item() for l in labels_list]
        else:
            stratify_labels = labels_list
        
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        list(range(len(dataset))),
        train_size=train_size,
        test_size=test_size,
        random_state=12345,
        shuffle=True,
        stratify=stratify_labels
    )
    
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Prepare Output Dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train Probes per Layer
    for layer in LAYERS:
        print(f"\n=== Training Probe for Layer {layer} ===")
        
        probe = LinearProbeClassification(
            device=device,
            probe_class=len(config["label_map"]),
            input_dim=5120, # Llama-2 13B dim
            logistic=True
        )
        
        train_config = TrainerConfig(learning_rate=1e-3, weight_decay=0.1)
        optimizer, scheduler = probe.configure_optimizers(train_config)
        
        # If one_hot=True, use BCELoss (as per notebook reference), otherwise CrossEntropy
        loss_func = nn.BCELoss() # Using BCELoss because logistic=True and one_hot=True
        
        best_acc = 0.0
        best_state = None
        
        for epoch in range(args.epochs):
            loss, acc = train(
                probe=probe,
                device=device,
                train_loader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                loss_func=loss_func,
                layer_num=layer - 1, 
                verbose=(epoch == args.epochs - 1),
                one_hot=True, # Pass one_hot to train/test
                num_classes=len(config["label_map"])
            )
            
            test_loss, test_acc = test( # test returns (loss, acc) by default
                probe=probe,
                device=device,
                test_loader=test_loader,
                loss_func=loss_func,
                layer_num=layer,
                one_hot=True,
                num_classes=len(config["label_map"])
            )
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_state = probe.state_dict()
                if epoch % 5 == 0:
                     print(f"Epoch {epoch}: New best accuracy {best_acc:.3f}")
        
        print(f"Layer {layer} Best Test Accuracy: {best_acc:.3f}")
        
        # Save Best Model
        if best_state is not None:
            save_path = os.path.join(args.output_dir, f"{config['probe_prefix']}_probe_at_layer_{layer}.pth")
            torch.save(best_state, save_path)
            print(f"Saved best probe to {save_path}")
        else:
             print("Warning: No best state found (accuracy 0?)")

    print("Training Complete!")

if __name__ == "__main__":
    main()
