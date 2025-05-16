import os
import sys
import torch
import numpy as np
import zipfile
from PIL import Image
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Import your trained model
from CLIP_image_distillation import CLIPImageDistillation

# Fix import issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Configuration ===
checkpoint_path = "C:/Users/Daniel Csizmadia/Desktop/zero_shot_graph/GRAPH_EPOCHS_BASE16_epoch-epoch=07-train_loss=1.99.ckpt"
imagenet_zip_path = "C:/Users/Daniel Csizmadia/Downloads/imagenet1k.zip"
extract_folder = "C:/Users/Daniel Csizmadia/Downloads/imagenet_val_extracted" 

imagenet_classes_folder = "C:/Users/Daniel Csizmadia/Downloads/imagenet_val_extracted/imagenet-val"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def extract_imagenet(zip_path, extract_to):
    """Extracts ImageNet validation dataset if not already extracted."""
    if not os.path.exists(extract_to):
        print("🔄 Extracting ImageNet validation dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Extraction Complete!")

def evaluate_zero_shot(model_name, model, processor, dataloader, classnames):
    """Evaluate a single model for zero-shot classification"""
    print(f"\n🔍 Evaluating {model_name}...")
    
    # Create prompts with template
    prompts = [f"a photo of a {name}" for name in classnames]
    
    # Process text prompts
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        
        if model_name == "base":
            text_features = model.get_text_features(**text_inputs)
        else:
            text_features = model.student.get_text_features(**text_inputs)
            
        text_features = F.normalize(text_features, dim=1)
    
    # Track metrics
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    # Process images in batches
    for images, labels in tqdm(dataloader, desc=f"Processing {model_name}"):
        with torch.no_grad():
            try:
                # Convert images to device
                images = images.to(device)
                
                # Apply exact CLIP normalization
                image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
                image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
                normalized_images = (images - image_mean) / image_std
                
                # Get image features based on model type
                if model_name == "base":
                    image_features = model.get_image_features(pixel_values=normalized_images)
                else:
                    image_features = model.student.get_image_features(pixel_values=normalized_images)
                
                # Normalize embeddings
                image_features = F.normalize(image_features, dim=1)
                
                # Calculate similarity scores with temperature
                similarity = 100.0 * (image_features @ text_features.T)
                
                # Get predictions
                values, indices = similarity.topk(5, dim=1)
                
                # Debug first batch
                if total == 0:
                    print(f"\n{model_name} - First 3 examples:")
                    for i in range(min(3, len(labels))):
                        true_label = labels[i].item()
                        pred_label = indices[i, 0].item()
                        print(f"True: {true_label} ({classnames[true_label] if true_label < len(classnames) else 'unknown'})")
                        print(f"Pred: {pred_label} ({classnames[pred_label] if pred_label < len(classnames) else 'unknown'})")
                        print(f"Top-5 preds: {indices[i].tolist()}")
                        print(f"Top-5 scores: {values[i].tolist()}")
                        print("---")
                
                # Count correct predictions
                top1_correct = (indices[:, 0] == labels.to(device)).sum().item()
                
                # For top-5, check if true label is in any of the top 5 predictions
                for i, label in enumerate(labels.to(device)):
                    if label in indices[i]:
                        correct_top5 += 1
                
                correct_top1 += top1_correct
                total += len(labels)
                
                # Print results every 1000 examples
                if total % 1000 == 0:
                    interim_top1 = correct_top1 / total
                    interim_top5 = correct_top5 / total
                    print(f"\nAfter {total} examples - {model_name} Top-1: {interim_top1:.4f}, Top-5: {interim_top5:.4f}")
                
            except Exception as e:
                print(f"Error processing batch: {e}")
    
    # Calculate accuracy
    top1 = correct_top1 / total if total > 0 else 0
    top5 = correct_top5 / total if total > 0 else 0
    
    print(f"{model_name} Top-1: {top1:.4f}, Top-5: {top5:.4f}")
    
    return {"top1": top1, "top5": top5}

def main():
    try:
        # Extract ImageNet dataset if necessary
        extract_imagenet(imagenet_zip_path, extract_folder)
        
        # Load ImageNet class names
        with open("C:/Users/Daniel Csizmadia/Downloads/imagenet_classes.txt", "r") as f:
            classnames = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(classnames)} ImageNet classes")
        print(f"First 5 classes: {classnames[:5]}")
        print(f"Last class: {classnames[-1]}")
        
        # Define transforms (without normalization - let processor handle it)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),  # Add center crop for consistency
            transforms.ToTensor()
        ])
        
        # Load dataset
        print("Loading ImageNet dataset...")
        imagenet_dataset = datasets.ImageFolder(imagenet_classes_folder, transform=transform)
        print(f"Dataset contains {len(imagenet_dataset)} images with {len(imagenet_dataset.classes)} classes")
        
        # Show dataset class mapping
        print(f"Dataset class mapping (first 5): {list(imagenet_dataset.class_to_idx.items())[:5]}")
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            imagenet_dataset, batch_size=1, shuffle=False)
        
        # Load models
        print("Loading models...")
        base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval() #openai/clip-vit-large-patch14 #openai/clip-vit-base-patch16
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # Evaluate base CLIP model
        #base_results = evaluate_zero_shot("base", base_model, clip_processor, dataloader, classnames)
        
        # Load and evaluate your custom model
        try:
            print("\nLoading custom model...")
            custom_model = CLIPImageDistillation.load_from_checkpoint(
                checkpoint_path,
                map_location=device,
                clip_model=base_model,
                clip_preprocess=clip_processor,
                strict = False
            ).to(device).eval()
            
            custom_results = evaluate_zero_shot("custom", custom_model, clip_processor, dataloader, classnames)
        except Exception as e:
            print(f"Error loading custom model: {e}")
            custom_results = {"top1": 0, "top5": 0}
        
        # Display results
        print("\n📊 Zero-Shot ImageNet Results")
        print("=" * 50)
        print(f"{'Model':<20} {'Top-1 Acc':<15} {'Top-5 Acc':<15}")
        print("-" * 50)
        #print(f"Base CLIP {base_results['top1']:.4f} {base_results['top5']:.4f}")
        print(f"Custom Model {custom_results['top1']:.4f} {custom_results['top5']:.4f}")
        
        # Calculate relative change if base model has non-zero performance
        #if base_results['top1'] > 0:
       #     rel_improvement = ((custom_results['top1'] - base_results['top1']) / 
       #                      base_results['top1']) * 100
       #     print(f"\nRelative improvement: {rel_improvement:+.2f}%")
        
        # Save results
        with open("imagenet_zero_shot_results.txt", "w") as f:
            f.write(f"Zero-Shot ImageNet Results\n")
           # f.write(f"Base CLIP Top-1: {base_results['top1']:.4f}\n")
            #f.write(f"Base CLIP Top-5: {base_results['top5']:.4f}\n\n")
            f.write(f"Custom Model Top-1: {custom_results['top1']:.4f}\n")
            f.write(f"Custom Model Top-5: {custom_results['top5']:.4f}\n\n")
            
            #if base_results['top1'] > 0:
                #f.write(f"Relative Top-1 improvement: {rel_improvement:+.2f}%\n")
    
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()