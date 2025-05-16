import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pickle
import json
import hashlib
import numpy as np
from PIL import Image
from image_tokenizer import CLIPPatchTokenizer, TokenizerWithKNN
import torch

# Modified functions to handle multiple captions per image format

def precache_yolo(json_file, cache_dir="./cache", mini_batch_size=32):
    """
    Creates the YOLO detection cache for the datasets
    Works with both old format (one caption per image) and new format (multiple captions per image)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {}
        
    # Extract unique image paths - handles both old and new format
    image_paths = []
    image_path_set = set()
    
    for item in data:
        if "image_path" in item:
            path = item["image_path"]
            if path and path not in image_path_set:
                image_path_set.add(path)
                image_paths.append(path)
    
    print(f"Found {len(image_paths)} unique images for YOLO detection")
    
    # Filter out non-existent images
    valid_paths = []
    for path in image_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"Warning: Image not found: {path}")
    
    print(f"Processing {len(valid_paths)} existing images")
    
    tokenizer = CLIPPatchTokenizer()
    cached_yolo = tokenizer.get_weighted_bounding_boxes_batch(valid_paths, mini_batch_size=mini_batch_size)
    
    precache_file = os.path.join(cache_dir, os.path.basename(json_file).replace('.json', '') + "_precache.pkl")
    with open(precache_file, 'wb') as f:
        pickle.dump(cached_yolo, f)
    print("YOLO precache saved to:", precache_file)
    
    return cached_yolo

def precache_knn(json_file, yolo_cache=None, cache_dir="./cache", 
                projection_model_path=None, 
                faiss_index_path=None,
                embeddings_json_path=None,
                similarity_threshold=0.85):
    """
    Creates a cache of KNN/projection results for faster inference
    Compatible with both old format (one caption per image) and new format (multiple captions per image)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    if yolo_cache is None:
        # Load the YOLO cache if not provided
        precache_file = os.path.join(cache_dir, os.path.basename(json_file).replace('.json', '') + "_precache.pkl")
        if os.path.exists(precache_file):
            with open(precache_file, 'rb') as f:
                yolo_cache = pickle.load(f)
        else:
            print(f"No YOLO cache found at {precache_file}. Run precache_yolo first.")
            return {}
    
    # Initialize advanced tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        advanced_tokenizer = TokenizerWithKNN(
            projection_model_path=projection_model_path,
            faiss_index_path=faiss_index_path,
            embeddings_json_path=embeddings_json_path,
            similarity_threshold=similarity_threshold,
            device=device
        )
    except Exception as e:
        print(f"Error initializing TokenizerWithKNN: {e}")
        return {}
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return {}
    
    # Extract unique image paths - handles both old and new format
    image_path_set = set()
    for item in data:
        if "image_path" in item:
            path = item["image_path"]
            if path:
                image_path_set.add(path)
                
    image_paths = list(image_path_set)
    print(f"Found {len(image_paths)} unique images for KNN processing")
    
    # Check which images are in YOLO cache
    images_in_cache = [path for path in image_paths if path in yolo_cache]
    print(f"{len(images_in_cache)} of {len(image_paths)} images found in YOLO cache")
    
    knn_cache = {}
    processed = 0
    
    # Process each image and its patches
    for image_path in images_in_cache:
        try:
            image = Image.open(image_path).convert("RGB")
            weighted_boxes = yolo_cache[image_path]
            
            for (box, conf) in weighted_boxes:
                x1, y1, x2, y2 = box
                # Normalize box coordinates
                w, h = image.size
                position = [x1/w, y1/h, x2/w, y2/h]
                
                # Extract patch
                patch = image.crop((x1, y1, x2, y2))
                
                # Create a cache key
                patch_bytes = np.array(patch).tobytes()[:1000]  # First 1000 bytes for efficiency
                position_str = f"{x1/w:.4f}_{y1/h:.4f}_{x2/w:.4f}_{y2/h:.4f}"
                cache_key = hashlib.md5(patch_bytes + position_str.encode()).hexdigest()
                
                # Get embedding
                try:
                    embedding, source, similarity = advanced_tokenizer.knn_or_projection(patch, position)
                    
                    # Serialize the embedding if needed
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    
                    # Store in cache
                    knn_cache[cache_key] = {
                        'embedding': embedding,
                        'source': source,
                        'similarity': similarity,
                        'image_path': image_path,
                        'position': position
                    }
                    
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed} patches...")
                        
                except Exception as e:
                    print(f"Error processing {image_path} patch: {e}")
        except Exception as e:
            print(f"Error opening {image_path}: {e}")
    
    # Save KNN cache
    cache_name = os.path.basename(json_file).replace('.json', '') + "_knn_cache.pkl"
    knn_cache_file = os.path.join(cache_dir, cache_name)
    with open(knn_cache_file, 'wb') as f:
        pickle.dump(knn_cache, f)
    print(f"KNN cache saved to {knn_cache_file} with {processed} entries")
    
    return knn_cache

if __name__ == "__main__":
    # Path to your new teacher dataset
    json_file = "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/teacher_dataset/teacher_10k_val.json"
    
    # Create output directory
    cache_dir = "./teacher_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # First create YOLO cache
    print("Creating YOLO detection cache...")
    yolo_cache = precache_yolo(json_file, cache_dir=cache_dir)
    
    # Then create KNN cache using the YOLO cache
    print("\nCreating KNN embedding cache...")
    precache_knn(json_file, 
                yolo_cache=yolo_cache,
                cache_dir=cache_dir,
                projection_model_path="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/ImageProjectionModuleDev/TrainedProjectionModule/PLACEHOLDER", #NewTrainedProjectionModule/proj_module_best.pth
                faiss_index_path="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/trained_models/faiss_clip_index.idx",
                embeddings_json_path="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/trained_models/clip_embeddings.json")
                
    print("Caching complete!")

    #REMEBER TO CHANGE THE NAMES OF THE CACHES SO IT WORKS