import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel


# Import your custom model
from CLIP_image_distillation import CLIPImageDistillation

from SFTDefaultCLIP import CLIPSimpleFineTune

def calculate_retrieval_metrics(similarity_matrix, image_ids, caption_image_ids):
    """Calculate metrics in a more memory-efficient way"""
    # Group captions by image ID for image-to-text evaluation
    img_to_captions = defaultdict(list)
    for i, img_id in enumerate(caption_image_ids):
        img_to_captions[img_id].append(i)
    
    # Text-to-Image retrieval (batched for efficiency)
    print("Calculating Text→Image metrics...")
    t2i_ranks = []
    
    # Process in small batches to reduce memory usage
    batch_size = 100
    for batch_start in range(0, len(caption_image_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(caption_image_ids))
        
        for caption_idx in range(batch_start, batch_end):
            gt_img_id = caption_image_ids[caption_idx]
            gt_img_idx = image_ids.index(gt_img_id)
            
            # Get just this caption's similarities
            similarities = similarity_matrix[caption_idx]
            
            # Find rank of ground truth image (more efficient approach)
            sorted_indices = np.argsort(-similarities)
            rank = np.where(sorted_indices == gt_img_idx)[0][0]
            t2i_ranks.append(rank)
    
    # Image-to-Text retrieval (batched for efficiency)
    print("Calculating Image→Text metrics...")
    i2t_ranks = []
    
    for batch_start in range(0, len(image_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(image_ids))
        
        for img_idx in range(batch_start, batch_end):
            img_id = image_ids[img_idx]
            gt_caption_indices = img_to_captions.get(img_id, [])
            
            if not gt_caption_indices:
                continue
                
            # Get just this image's similarities to captions
            similarities = similarity_matrix[:, img_idx]
            
            # Find best rank among ground truth captions
            sorted_indices = np.argsort(-similarities)
            best_rank = min([np.where(sorted_indices == gt_idx)[0][0] for gt_idx in gt_caption_indices])
            i2t_ranks.append(best_rank)
    
    # Calculate metrics
    def recall_at_k(ranks, k):
        return len([r for r in ranks if r < k]) / len(ranks)
    
    def mean_ap(ranks):
        return np.mean([1.0 / (r + 1) for r in ranks])
    
    metrics = {
        "t2i": {
            "R@1": recall_at_k(t2i_ranks, 1),
            "R@5": recall_at_k(t2i_ranks, 5),
            "R@10": recall_at_k(t2i_ranks, 10),
            "MAP": mean_ap(t2i_ranks)
        },
        "i2t": {
            "R@1": recall_at_k(i2t_ranks, 1),
            "R@5": recall_at_k(i2t_ranks, 5),
            "R@10": recall_at_k(i2t_ranks, 10),
            "MAP": mean_ap(i2t_ranks)
        }
    }
    
    return metrics

def evaluate_model(model_name, device, max_images=1000):
    """Evaluate model on Flickr30K using batching for efficiency"""
    print(f"\n=== Evaluating {model_name} Model ===")
    
    # === Load Dataset ===
    DATASET_JSON = "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/flickr30k_test_karpathy.json" #coco_flickr_format.json #flickr30k_standard.json
    with open(DATASET_JSON, "r") as f: #flickr30k_test_karpathy.json #coco_test_karpathy.json
        dataset = json.load(f)
    
    # Filter out entries with missing captions and limit dataset size
    dataset = [item for item in dataset if item.get("captions") and len(item.get("captions")) > 0]
    
    if max_images > 0 and max_images < len(dataset):
        print(f"Limiting evaluation to first {max_images} images")
        dataset = dataset[:max_images]
    
    print(f"Testing on {len(dataset)} images with captions")
    
    # === Set up Models ===
    base_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device) #openai/clip-vit-base-patch16 #openai/clip-vit-large-patch14
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    if model_name == "custom":
        checkpoint_path = "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/models/checkpoints/epoch-epoch=01-train_loss=3.01.ckpt" #C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/models/checkpoints/epoch-epoch=04-train_loss=1.81.ckpt 
        try:
             # Use the correct model class based on the checkpoint path
            if "sft_baseline" in checkpoint_path:
                model = CLIPSimpleFineTune.load_from_checkpoint(
                    checkpoint_path,
                    map_location=device,
                    clip_model=base_clip_model,
                    clip_preprocess=clip_processor
                ).to(device)
                # Use the model's underlying CLIP model when evaluating
                model = model.model  # This gives you the actual CLIP model
            else:
                model = CLIPImageDistillation.load_from_checkpoint(
                    checkpoint_path,
                    map_location=device,
                    clip_model=base_clip_model,
                    clip_preprocess=clip_processor,
                    strict=False  
                ).to(device)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Using default path...")
            checkpoint_path = "C:/Users/Daniel Csizmadia/Desktop/new_checkpoints/epoch-epoch=00-train_loss=1.18.ckpt"
            model = CLIPImageDistillation.load_from_checkpoint(
                checkpoint_path,
                map_location=device,
                clip_model=base_clip_model,
                clip_preprocess=clip_processor,
                strict=False  # Add this parameter here
            ).to(device)
    else:
        model = base_clip_model
    
    model.eval()
    
    # === Extract Embeddings with Batching ===
    image_embeddings = []
    image_ids = []
    caption_embeddings = []
    caption_image_ids = []
    
    # Process images in batches
    print("Processing images...")
    batch_size = 4  # Adjust based on your GPU memory
    for i in range(0, len(dataset), batch_size):
        batch_items = dataset[i:i+batch_size]
        batch_images = []
        batch_paths = []
        
        for item in batch_items:
            try:
                image_path = item["image_path"]
                image = Image.open(image_path).convert("RGB")
                batch_images.append(image)
                batch_paths.append(item["image_id"])
            except Exception as e:
                print(f"Error loading image {item['image_path']}: {e}")
        
        if not batch_images:
            continue
            
        # Process batch
        with torch.no_grad():
            if model_name == "custom":
                # Process differently based on model type
                if "sft_baseline" in checkpoint_path:
                    # For SFT model (base CLIP), use processor
                    inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True).to(device)
                    embeddings = model.get_image_features(**inputs).cpu().numpy()
                    image_embeddings.extend(embeddings)
                    image_ids.extend(batch_paths)
                else:
                    # For your custom model
                    for img, img_id in zip(batch_images, batch_paths):
                        embedding = model(image=img).cpu().numpy()
                        image_embeddings.append(embedding[0])
                        image_ids.append(img_id)
            else:
                # Process as batch for base CLIP
                inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True).to(device)
                embeddings = model.get_image_features(**inputs).cpu().numpy()
                image_embeddings.extend(embeddings)
                image_ids.extend(batch_paths)
    
    # Process captions in batches
    print("Processing captions...")
    for i in range(0, len(dataset), batch_size):
        batch_items = dataset[i:i+batch_size]
        
        for item in batch_items:
            item_captions = []
            item_ids = []
            
            for caption in item["captions"]:
                item_captions.append(caption)
                item_ids.append(item["image_id"])
            
            # Process batch of captions
            with torch.no_grad():
                if model_name == "custom":
                    # Process one by one for custom model
                    if "sft_baseline" in checkpoint_path:
                        # For SFT model (base CLIP), use processor
                        try:
                            inputs = clip_processor(text=item_captions, return_tensors="pt", padding=True).to(device)
                            embeddings = model.get_text_features(**inputs).cpu().numpy()
                            caption_embeddings.extend(embeddings)
                            caption_image_ids.extend(item_ids)
                        except Exception as e:
                            print(f"Error processing caption batch: {e}")
                    else:
                        # For your custom model
                        for caption, img_id in zip(item_captions, item_ids):
                            try:
                                embedding = model(text=caption).cpu().numpy()
                                caption_embeddings.append(embedding[0])
                                caption_image_ids.append(img_id)
                            except Exception as e:
                                print(f"Error processing caption: {e}")
                else:
                    # Process as batch for base CLIP
                    try:
                        inputs = clip_processor(text=item_captions, return_tensors="pt", padding=True).to(device)
                        embeddings = model.get_text_features(**inputs).cpu().numpy()
                        caption_embeddings.extend(embeddings)
                        caption_image_ids.extend(item_ids)
                    except Exception as e:
                        print(f"Error processing caption batch: {e}")
    
    # Convert to arrays and normalize
    print(f"Computing metrics for {len(image_embeddings)} images and {len(caption_embeddings)} captions")
    image_embeddings = np.array(image_embeddings)
    caption_embeddings = np.array(caption_embeddings)
    
    # Normalize embeddings
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True)
    
    # Calculate similarity matrix (compute in chunks to save memory)
    print("Computing similarity matrix...")
    chunk_size = 1000
    similarity_matrix = np.zeros((len(caption_embeddings), len(image_embeddings)))
    
    for i in range(0, len(caption_embeddings), chunk_size):
        end_i = min(i + chunk_size, len(caption_embeddings))
        caption_chunk = caption_embeddings[i:end_i]
        
        for j in range(0, len(image_embeddings), chunk_size):
            end_j = min(j + chunk_size, len(image_embeddings))
            image_chunk = image_embeddings[j:end_j]
            
            # Calculate chunk of similarity matrix
            similarity_matrix[i:end_i, j:end_j] = np.matmul(caption_chunk, image_chunk.T)
    
    # Calculate metrics
    metrics = calculate_retrieval_metrics(similarity_matrix, image_ids, caption_image_ids)
    
    # Print results
    print("\n--- Text-to-Image Retrieval ---")
    print(f"Recall@1: {metrics['t2i']['R@1']:.4f}")
    print(f"Recall@5: {metrics['t2i']['R@5']:.4f}")
    print(f"Recall@10: {metrics['t2i']['R@10']:.4f}")
    print(f"MAP: {metrics['t2i']['MAP']:.4f}")
    
    print("\n--- Image-to-Text Retrieval ---")
    print(f"Recall@1: {metrics['i2t']['R@1']:.4f}")
    print(f"Recall@5: {metrics['i2t']['R@5']:.4f}")
    print(f"Recall@10: {metrics['i2t']['R@10']:.4f}")
    print(f"MAP: {metrics['i2t']['MAP']:.4f}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate models on Flickr30K')
    parser.add_argument('--max_images', type=int, default=1000, 
                        help='Maximum number of images to evaluate (default: 1000)')
    parser.add_argument('--model', type=str, default='both',
                        choices=['base', 'custom', 'both'],
                        help='Which model(s) to evaluate (default: both)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to custom model checkpoint')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Evaluate selected models
    base_results = None
    custom_results = None
    
    if args.model in ['base', 'both']:
        base_results = evaluate_model("base", device, max_images=args.max_images)
    
    if args.model in ['custom', 'both']:
        custom_results = evaluate_model("custom", device, max_images=args.max_images)
    
    # Print comparison if both models were evaluated
    if base_results and custom_results:
        print("\n=== Model Comparison ===")
        print("                Base CLIP    Custom Model")
        print(f"T→I Recall@1:    {base_results['t2i']['R@1']:.4f}        {custom_results['t2i']['R@1']:.4f}")
        print(f"T→I Recall@5:    {base_results['t2i']['R@5']:.4f}        {custom_results['t2i']['R@5']:.4f}")
        print(f"T→I Recall@10:   {base_results['t2i']['R@10']:.4f}        {custom_results['t2i']['R@10']:.4f}")
        print(f"T→I MAP:         {base_results['t2i']['MAP']:.4f}        {custom_results['t2i']['MAP']:.4f}")
        print(f"I→T Recall@1:    {base_results['i2t']['R@1']:.4f}        {custom_results['i2t']['R@1']:.4f}")
        print(f"I→T Recall@5:    {base_results['i2t']['R@5']:.4f}        {custom_results['i2t']['R@5']:.4f}")
        print(f"I→T Recall@10:   {base_results['i2t']['R@10']:.4f}        {custom_results['i2t']['R@10']:.4f}")
        print(f"I→T MAP:         {base_results['i2t']['MAP']:.4f}        {custom_results['i2t']['MAP']:.4f}")
        
        # Calculate relative improvements
        t2i_r1_gain = (custom_results['t2i']['R@1'] - base_results['t2i']['R@1']) / base_results['t2i']['R@1'] * 100
        i2t_r1_gain = (custom_results['i2t']['R@1'] - base_results['i2t']['R@1']) / base_results['i2t']['R@1'] * 100
        
        print(f"\nRelative Gains:")
        print(f"Text→Image R@1: +{t2i_r1_gain:.1f}%")
        print(f"Image→Text R@1: +{i2t_r1_gain:.1f}%")