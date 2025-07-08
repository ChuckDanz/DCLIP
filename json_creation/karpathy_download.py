import os
import json
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse

def download_karpathy_split(dataset_name):
    """Download the Karpathy split files for a specific dataset
    
    Args:
        dataset_name: Either 'flickr30k' or 'coco'
    """
    # Set URL based on dataset name
    if dataset_name == "flickr30k":
        url = 'https://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip'
    elif dataset_name == "coco":
        url = 'https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Must be 'flickr30k' or 'coco'")
    
    # Create directories
    data_dir = Path("C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/data/karpathy")
    data_dir.mkdir(exist_ok=True, parents=True)
    
    zip_path = data_dir / f"{dataset_name}.zip"
    json_path = data_dir / dataset_name / f"dataset_{dataset_name}.json"
    
    # Download if not exists
    if not zip_path.exists():
        print(f"Downloading {dataset_name} Karpathy split...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(zip_path, 'wb') as f, tqdm(
                desc=f"Downloading {dataset_name}",
                total=total_size,
                unit='B',
                unit_scale=True) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
    else:
        print(f"{dataset_name} Karpathy split zip already exists at {zip_path}")
    
    # Extract if not already extracted
    if not json_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Extracted to {data_dir}")
    else:
        print(f"Karpathy split already extracted at {json_path}")
    
    return json_path

def prepare_flickr30k_karpathy_json(image_dir, karpathy_json_path, output_json, split="test"):
    """
    Creates a Flickr30k JSON file using the Karpathy split
    
    Args:
        image_dir: Directory containing Flickr30k images
        karpathy_json_path: Path to the Karpathy JSON dataset file
        output_json: Path to output the JSON file
        split: Which split to extract ('train', 'val', 'test')
    """
    # Load Karpathy split data
    print(f"Reading Karpathy split from {karpathy_json_path}")
    with open(karpathy_json_path, 'r', encoding='utf-8') as f:
        karpathy_data = json.load(f)
    
    # Filter images by split and prepare the output format
    flickr_data = []
    image_count = 0
    caption_count = 0
    images_not_found = 0
    
    print(f"Processing {split} split from Karpathy's data...")
    for img in karpathy_data['images']:
        # Only include images from the specified split
        if img['split'] != split:
            continue
            
        image_name = img['filename']
        image_path = os.path.join(image_dir, image_name)
        
        # Check if image exists
        if not os.path.exists(image_path):
            images_not_found += 1
            if images_not_found <= 5:
                print(f"Warning: Image not found: {image_path}")
            continue
        
        # Extract captions
        captions = [sent['raw'] for sent in img['sentences']]
        
        # Create entry in the desired format
        entry = {
            "image_path": image_path,
            "image_id": img['imgid'],  # Use the original image ID from Karpathy
            "captions": captions
        }
        
        flickr_data.append(entry)
        image_count += 1
        caption_count += len(captions)
    
    if images_not_found > 5:
        print(f"... and {images_not_found - 5} more missing images")
    
    # Write to JSON file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(flickr_data, f, indent=2)
    
    print(f"Created Flickr30k {split} split JSON with {image_count} images and {caption_count} captions")
    expected_counts = {"test": 1000, "val": 1000, "train": 29000}
    if split in expected_counts and image_count != expected_counts[split]:
        print(f"Warning: Expected {expected_counts[split]} images for {split} split, but found {image_count}")
    
    return flickr_data

def prepare_coco_karpathy_json(coco_dir, karpathy_json_path, output_json, split="test"):
    """
    Creates a MSCOCO JSON file using the Karpathy split
    
    Args:
        coco_dir: Directory containing MSCOCO base directory
        karpathy_json_path: Path to the Karpathy JSON dataset file
        output_json: Path to output the JSON file
        split: Which split to extract ('train', 'val', 'test')
    """
    # Load Karpathy split data
    print(f"Reading Karpathy split from {karpathy_json_path}")
    with open(karpathy_json_path, 'r', encoding='utf-8') as f:
        karpathy_data = json.load(f)
    
    # Filter images by split and prepare the output format
    coco_data = []
    image_count = 0
    caption_count = 0
    images_not_found = 0
    
    print(f"Processing {split} split from Karpathy's data...")
    for img in karpathy_data['images']:
        # Only include images from the specified split
        if img['split'] != split:
            continue
            
        # Determine the correct subdirectory based on filename prefix
        if 'COCO_train2014_' in img['filename']:
            subdir = 'train2014'
        elif 'COCO_val2014_' in img['filename']:
            subdir = 'val2014'
        else:
            print(f"Unknown image format: {img['filename']}, skipping...")
            continue
            
        image_path = os.path.join(coco_dir, subdir, img['filename'])
        
        # Check if image exists
        if not os.path.exists(image_path):
            images_not_found += 1
            if images_not_found <= 5:
                print(f"Warning: Image not found: {image_path}")
            continue
        
        # Extract captions
        captions = [sent['raw'] for sent in img['sentences']]
        
        # Create entry in the desired format
        entry = {
            "image_path": image_path,
            "image_id": img['imgid'],  # Use the original image ID from Karpathy
            "captions": captions
        }
        
        coco_data.append(entry)
        image_count += 1
        caption_count += len(captions)
    
    if images_not_found > 5:
        print(f"... and {images_not_found - 5} more missing images")
    
    # Write to JSON file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Created COCO {split} split JSON with {image_count} images and {caption_count} captions")
    expected_counts = {"test": 5000, "val": 5000, "train": 113287, "restval": 30504}
    if split in expected_counts and image_count != expected_counts[split]:
        print(f"Warning: Expected {expected_counts[split]} images for {split} split, but found {image_count}")
    
    return coco_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare Karpathy split JSONs for MSCOCO and/or Flickr30k')
    
    # Dataset selection
    parser.add_argument('--datasets', 
                      choices=['coco', 'flickr30k', 'both'],
                      default='both',
                      help='Which dataset(s) to process')
    
    # Directory paths
    parser.add_argument('--coco_dir', 
                      default="C:/Users/Daniel Csizmadia/Desktop/coco2014",
                      help='Directory containing MSCOCO (should have train2014 and val2014 subdirs)')
    parser.add_argument('--flickr_dir', 
                      default="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/flickr30k_images/flickr30k_images",
                      help='Directory containing Flickr30K images')
    parser.add_argument('--output_dir',
                      default="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP",
                      help='Directory to save output JSON files')
    
    # Split selection
    parser.add_argument('--split', 
                      default="all", 
                      help='Which split to use. "all" processes all splits. For COCO: train, val, test, restval. For Flickr30k: train, val, test')
    
    args = parser.parse_args()
    
    # Print configuration
    print("========== CONFIGURATION ==========")
    print(f"Processing datasets: {args.datasets}")
    if args.datasets in ['coco', 'both']:
        print(f"MSCOCO directory: {args.coco_dir}")
    if args.datasets in ['flickr30k', 'both']:
        print(f"Flickr30k directory: {args.flickr_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split: {args.split}")
    print("=================================")
    
    # Process MSCOCO if requested
    if args.datasets in ['coco', 'both']:
        print("\n===== PROCESSING MSCOCO =====")
        karpathy_json_path = download_karpathy_split("coco")
        
        # Determine splits to process
        if args.split == "all":
            splits = ["train", "val", "test", "restval"]
        else:
            splits = [args.split]
        
        for split in splits:
            output_json = os.path.join(args.output_dir, f"coco_{split}_karpathy.json")
            prepare_coco_karpathy_json(args.coco_dir, karpathy_json_path, output_json, split)
    
    # Process Flickr30k if requested
    if args.datasets in ['flickr30k', 'both']:
        print("\n===== PROCESSING FLICKR30K =====")
        karpathy_json_path = download_karpathy_split("flickr30k")
        
        # Determine splits to process
        if args.split == "all":
            splits = ["train", "val", "test"]
        else:
            splits = [args.split]
        
        for split in splits:
            output_json = os.path.join(args.output_dir, f"flickr30k_{split}_karpathy.json")
            prepare_flickr30k_karpathy_json(args.flickr_dir, karpathy_json_path, output_json, split)
    
    print("\nAll processing completed successfully!")