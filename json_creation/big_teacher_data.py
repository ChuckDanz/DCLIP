import os
import json
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import argparse

class DatasetPreparation:
    def __init__(self, 
                 output_dir="./combined_dataset",
                 coco_images_dir=None, 
                 coco_annotations_file=None,
                 vg_images_dir=None, 
                 vg_annotations_file=None,
                 flickr_images_dir=None, 
                 flickr_annotations_file=None,
                 cc_images_dir=None, 
                 cc_annotations_file=None):
        """Initialize with explicit paths for all datasets"""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.json_path = os.path.join(output_dir, "vg_test_train.json")
        self.val_json_path = os.path.join(output_dir, "vg_test_val.json")
        
        # Store explicit paths
        self.coco_images_dir = coco_images_dir
        self.coco_annotations_file = coco_annotations_file
        
        self.vg_images_dir = vg_images_dir
        self.vg_annotations_file = vg_annotations_file
        
        self.flickr_images_dir = flickr_images_dir
        self.flickr_annotations_file = flickr_annotations_file
        
        self.cc_images_dir = cc_images_dir
        self.cc_annotations_file = cc_annotations_file
        
        # Set target counts for each dataset
        self.targets = {
            "coco": 50000,
            "visual_genome": 25000,
            "flickr30k": 15000, 
            "conceptual_captions": 10000
        }
    
    def process_coco(self, target_count=50000):
        """Process MSCOCO dataset using explicit paths"""
        if not self.coco_images_dir or not self.coco_annotations_file:
            print("Skipping MSCOCO: path not provided")
            return []
            
        if not os.path.exists(self.coco_images_dir) or not os.path.exists(self.coco_annotations_file):
            print(f"COCO directory or annotations file not found. Skipping.")
            return []
            
        print(f"Processing MSCOCO dataset (target: {target_count} images)...")
        print(f"- Images directory: {self.coco_images_dir}")
        print(f"- Annotations file: {self.coco_annotations_file}")
        
        # Load annotations
        with open(self.coco_annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Group captions by image id
        images_by_id = {}
        for img in coco_data['images']:
            images_by_id[img['id']] = {
                'file_name': img['file_name'],
                'captions': []
            }
        
        for caption in coco_data['annotations']:
            if caption['image_id'] in images_by_id:
                images_by_id[caption['image_id']]['captions'].append(caption['caption'])
        
        # Convert to our format
        results = []
        
        for img_id, img_data in tqdm(list(images_by_id.items())[:target_count*2]):  # Process more to account for missing files
            image_path = os.path.join(self.coco_images_dir, img_data['file_name'])
            if not os.path.exists(image_path):
                continue
                
            # Add to results
            results.append({
                "image_path": image_path,
                "captions": img_data['captions'],
                "dataset": "coco",
                "boxes": []  # No box information for basic COCO
            })
            
            # Check if we've hit our target
            if len(results) >= target_count:
                break
        
        print(f"Processed {len(results)} MSCOCO images with {sum(len(item['captions']) for item in results)} captions")
        return results

    def process_visual_genome(self, target_count=25000):
        """Process Visual Genome dataset using explicit paths"""
        if not self.vg_images_dir or not self.vg_annotations_file:
            print("Skipping Visual Genome: path not provided")
            return []
            
        if not os.path.exists(self.vg_images_dir) or not os.path.exists(self.vg_annotations_file):
            print(f"Visual Genome directory or annotations file not found. Skipping.")
            return []
            
        print(f"Processing Visual Genome dataset (target: {target_count} images)...")
        print(f"- Images directory: {self.vg_images_dir}")
        print(f"- Annotations file: {self.vg_annotations_file}")
        
        # Load region descriptions
        with open(self.vg_annotations_file, 'r') as f:
            vg_regions = json.load(f)
        
        results = []
        
        for image_data in tqdm(vg_regions[:target_count*2]):  # Process more to account for filtering
            image_id = image_data['id']
            image_path = os.path.join(self.vg_images_dir, f"{image_id}.jpg")
            
            if not os.path.exists(image_path):
                # Try alternative extensions
                for ext in ['png', 'jpeg']:
                    alt_path = os.path.join(self.vg_images_dir, f"{image_id}.{ext}")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
                else:
                    continue  # Skip if image not found
            
            # Collect captions and boxes from regions
            captions = []
            boxes = []
            
            for region in image_data['regions']:
                if 'phrase' in region:
                    captions.append(region['phrase'])
                    
                    # Extract box coordinates if available
                    if 'x' in region and 'y' in region and 'width' in region and 'height' in region:
                        boxes.append({
                            'x': region['x'],
                            'y': region['y'],
                            'width': region['width'],
                            'height': region['height']
                        })
            
            # Only include if we have captions
            if captions:
                results.append({
                    "image_path": image_path,
                    "captions": captions,
                    "dataset": "visual_genome",
                    "boxes": boxes
                })
            
            # Check if we've hit our target
            if len(results) >= target_count:
                break
        
        print(f"Processed {len(results)} Visual Genome images with {sum(len(item['captions']) for item in results)} captions")
        return results

    def process_flickr30k(self, target_count=15000):
        """Process Flickr30K dataset using explicit paths"""
        if not self.flickr_images_dir or not self.flickr_annotations_file:
            print("Skipping Flickr30K: path not provided")
            return []
            
        if not os.path.exists(self.flickr_images_dir) or not os.path.exists(self.flickr_annotations_file):
            print(f"Flickr30K directory or annotations file not found. Skipping.")
            return []
            
        print(f"Processing Flickr30K dataset (target: {target_count} images)...")
        print(f"- Images directory: {self.flickr_images_dir}")
        print(f"- Annotations file: {self.flickr_annotations_file}")
        
        # Group captions by image
        captions_by_image = defaultdict(list)
        
        with open(self.flickr_annotations_file, 'r', encoding='utf-8') as f:
            # Skip header if present
            first_line = f.readline().strip()
            if "image_name" in first_line and "comment" in first_line:
                header = True
            else:
                header = False
                # Process first line if it's not a header
                parts = first_line.split("|")
                if len(parts) >= 3:
                    image_name = parts[0].strip()
                    caption = parts[2].strip() 
                    captions_by_image[image_name].append(caption)
            
            # Process remaining lines
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    image_name = parts[0].strip()
                    caption = parts[2].strip()
                    captions_by_image[image_name].append(caption)
        
        # Convert to our format
        results = []
        for image_name, captions in tqdm(list(captions_by_image.items())[:target_count*2]):
            image_path = os.path.join(self.flickr_images_dir, image_name)
            
            if not os.path.exists(image_path):
                continue
            
            results.append({
                "image_path": image_path,
                "captions": captions,
                "dataset": "flickr30k",
                "boxes": []  # No box information for Flickr30K
            })
            
            # Check if we've hit our target
            if len(results) >= target_count:
                break
        
        print(f"Processed {len(results)} Flickr30K images with {sum(len(item['captions']) for item in results)} captions")
        return results

    def process_conceptual_captions(self, target_count=10000):
        """Process Conceptual Captions dataset with on-the-fly image downloading"""
        if not self.cc_images_dir or not self.cc_annotations_file:
            print("Skipping Conceptual Captions: path not provided")
            return []
            
        if not os.path.exists(self.cc_annotations_file):
            print(f"Conceptual Captions annotations file not found. Skipping.")
            return []
            
        # Create images directory if it doesn't exist
        os.makedirs(self.cc_images_dir, exist_ok=True)
            
        print(f"Processing Conceptual Captions dataset (target: {target_count} images)...")
        print(f"- Images directory: {self.cc_images_dir}")
        print(f"- Annotations file: {self.cc_annotations_file}")
        
        # Import only when needed
        import requests
        from io import BytesIO
        
        results = []
        processed_count = 0
        skipped_count = 0
        download_count = 0
        
        # Set up HTTP headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
            
        with open(self.cc_annotations_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
            
        with open(self.cc_annotations_file, 'r', encoding='utf-8') as f:
            max_lines = min(total_lines, target_count * 5)  # Process 5x more than needed to account for failures
            
            for i, line in enumerate(tqdm(f, total=max_lines, desc="Processing Conceptual Captions")):
                if i >= max_lines:
                    break
                    
                if i == 0 and line.startswith("caption"):  # Skip header if present
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                    
                caption = parts[0].strip()
                image_url = parts[1].strip()
                
                if not image_url or not caption:
                    skipped_count += 1
                    continue
                
                # Create a filename from the URL
                filename = f"cc_{i:07d}_{image_url.split('/')[-1].split('?')[0]}"
                # Handle case of URLs without valid filename part
                if not filename or filename == f"cc_{i:07d}_":
                    filename = f"cc_{i:07d}.jpg"
                    
                # Clean up filename to remove invalid characters
                filename = "".join(c for c in filename if c.isalnum() or c in "._-")
                image_path = os.path.join(self.cc_images_dir, filename)
                
                # Skip if we already have this image
                if os.path.exists(image_path):
                    try:
                        with Image.open(image_path) as img:
                            # Image is valid, use it
                            results.append({
                                "image_path": image_path,
                                "captions": [caption],
                                "dataset": "conceptual_captions",
                                "boxes": []
                            })
                            processed_count += 1
                            
                            if processed_count >= target_count:
                                break
                            continue
                    except:
                        # Image exists but is invalid, will re-download
                        pass
                
                # Download the image
                try:
                    response = requests.get(image_url, stream=True, timeout=5, headers=headers)
                    if response.status_code == 200:
                        # Save image to disk
                        try:
                            # First validate it's a proper image
                            img = Image.open(BytesIO(response.content))
                            img.save(image_path)
                            
                            results.append({
                                "image_path": image_path,
                                "captions": [caption],
                                "dataset": "conceptual_captions",
                                "boxes": []
                            })
                            
                            processed_count += 1
                            download_count += 1
                            
                            # Print progress occasionally
                            if download_count % 100 == 0:
                                print(f"Downloaded {download_count} images so far")
                                
                            if processed_count >= target_count:
                                break
                        except Exception as e:
                            # Skip invalid images
                            skipped_count += 1
                except Exception as e:
                    # Skip on any error
                    skipped_count += 1
        
        print(f"Processed {len(results)} Conceptual Captions images")
        print(f"Downloaded {download_count} new images")
        print(f"Skipped {skipped_count} invalid or unreachable images")
        print(f"Total captions: {sum(len(item['captions']) for item in results)}")
        return results

    def combine_datasets(self):
        """Combine all datasets into a single JSON file"""
        print("\nCombining datasets...")
        
        all_data = []
        
        # Process each dataset using explicit paths
        coco_data = self.process_coco(self.targets["coco"])
        all_data.extend(coco_data)
        
        vg_data = self.process_visual_genome(self.targets["visual_genome"])
        all_data.extend(vg_data)
        
        flickr_data = self.process_flickr30k(self.targets["flickr30k"])
        all_data.extend(flickr_data)
        
        cc_data = self.process_conceptual_captions(self.targets["conceptual_captions"])
        all_data.extend(cc_data)
        
        if not all_data:
            print("Warning: No datasets were successfully processed!")
            return None, None
            
        # Shuffle data
        random.shuffle(all_data)
        
        # Split into train and validation (90/10 split)
        split_idx = int(len(all_data) * 0.9)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        # Save train and validation files
        print(f"Saving {len(train_data)} training examples to {self.json_path}")
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2)
            
        print(f"Saving {len(val_data)} validation examples to {self.val_json_path}")
        with open(self.val_json_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2)
            
        print(f"\nDataset preparation complete!")
        print(f"Training data: {self.json_path}")
        print(f"Validation data: {self.val_json_path}")
        
        # Print dataset statistics
        self.print_dataset_stats(train_data)
        
        return self.json_path, self.val_json_path
    
    def print_dataset_stats(self, data):
        """Print statistics about the combined dataset"""
        dataset_counts = {}
        caption_lengths = []
        images_with_boxes = 0
        
        for item in data:
            dataset = item.get("dataset", "unknown")
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
            
            for caption in item["captions"]:
                caption_lengths.append(len(caption.split()))
                
            if item.get("boxes") and len(item["boxes"]) > 0:
                images_with_boxes += 1
        
        print("\n=== Dataset Statistics ===")
        print(f"Total images: {len(data)}")
        print(f"Images with bounding boxes: {images_with_boxes} ({images_with_boxes/len(data)*100:.2f}%)")
        print("\nDistribution by dataset:")
        for dataset, count in dataset_counts.items():
            print(f"- {dataset}: {count} ({count/len(data)*100:.2f}%)")
        
        print("\nCaption statistics:")
        print(f"- Total captions: {sum(len(item['captions']) for item in data)}")
        print(f"- Avg captions per image: {sum(len(item['captions']) for item in data)/len(data):.2f}")
        avg_length = sum(caption_lengths) / len(caption_lengths)
        print(f"- Avg caption length: {avg_length:.2f} words")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare 100K dataset with manual paths")
    
    parser.add_argument('--output_dir', default="./combined_dataset",
                    help='Directory to save output JSON files')
    
    # COCO dataset paths
    parser.add_argument('--coco_images', 
                    help='Directory containing MSCOCO images (e.g., train2017 folder)')
    parser.add_argument('--coco_annotations', 
                    help='Path to MSCOCO annotations file (e.g., captions_train2017.json)')
    
    # Visual Genome dataset paths
    parser.add_argument('--vg_images', 
                    help='Directory containing Visual Genome images')
    parser.add_argument('--vg_annotations', 
                    help='Path to Visual Genome region descriptions file')
    
    # Flickr30K dataset paths
    parser.add_argument('--flickr_images', 
                    help='Directory containing Flickr30K images')
    parser.add_argument('--flickr_annotations', 
                    help='Path to Flickr30K results.csv file')
    
    # Conceptual Captions dataset paths
    parser.add_argument('--cc_images', 
                    help='Directory containing Conceptual Captions images')
    parser.add_argument('--cc_annotations', 
                    help='Path to Conceptual Captions TSV file')
    
    # Dataset targets (optional)
    parser.add_argument('--coco_target', type=int, default=50000,
                    help='Target number of COCO images')
    parser.add_argument('--vg_target', type=int, default=25000,
                    help='Target number of Visual Genome images')
    parser.add_argument('--flickr_target', type=int, default=15000,
                    help='Target number of Flickr30K images')
    parser.add_argument('--cc_target', type=int, default=10000,
                    help='Target number of Conceptual Captions images')
    
    args = parser.parse_args()
    
    print("=== 100K Teacher Dataset Preparation ===")
    print(f"Output directory: {args.output_dir}")
    
    # Update targets if specified
    targets = {
        "coco": args.coco_target,
        "visual_genome": args.vg_target,
        "flickr30k": args.flickr_target,
        "conceptual_captions": args.cc_target
    }
    
    # Display paths being used
    print("\nDataset paths:")
    if args.coco_images and args.coco_annotations:
        print(f"COCO: {args.coco_images} (target: {targets['coco']} images)")
    if args.vg_images and args.vg_annotations:
        print(f"Visual Genome: {args.vg_images} (target: {targets['visual_genome']} images)")
    if args.flickr_images and args.flickr_annotations:
        print(f"Flickr30K: {args.flickr_images} (target: {targets['flickr30k']} images)")
    if args.cc_images and args.cc_annotations:
        print(f"Conceptual Captions: {args.cc_images} (target: {targets['conceptual_captions']} images)")
    
    # Initialize and run preparation
    preparer = DatasetPreparation(
        output_dir=args.output_dir,
        coco_images_dir=args.coco_images,
        coco_annotations_file=args.coco_annotations,
        vg_images_dir=args.vg_images,
        vg_annotations_file=args.vg_annotations,
        flickr_images_dir=args.flickr_images,
        flickr_annotations_file=args.flickr_annotations,
        cc_images_dir=args.cc_images,
        cc_annotations_file=args.cc_annotations
    )
    
    # Update targets if needed
    preparer.targets = targets
    
    # Process datasets
    train_path, val_path = preparer.combine_datasets()
    
    if train_path:
        print("\n=== Next Steps ===")
        print(f"To train your teacher model, run:")
        print(f"python train_contrastive_teacher.py --train_file {train_path} --batch_size 64 --epochs 5 --output_path ./teacher_100k.pth")


        