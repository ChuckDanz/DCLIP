import os
import sys
import torch
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

# Import your trained model
from CLIP_image_distillation import CLIPImageDistillation

# Fix import issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Configuration ===
checkpoint_path = "PATH/TO/CUSTOM/MODEL/epoch-epoch=01-train_loss=1.01PAPERViT-L.ckpt"
data_dir = "./data"  # Where to store the CIFAR datasets
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def download_cifar_datasets(data_dir="./data"):
    """Download CIFAR-10 and CIFAR-100 datasets and prepare for zero-shot testing"""
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transform (resize to CLIP's expected input size)
    transform = transforms.Compose([
        transforms.Resize(224),  # CLIP expects 224x224 images
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    # Download CIFAR-10 (downloads automatically if not present)
    print("Downloading/loading CIFAR-10...")
    cifar10 = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    cifar10_classes = cifar10.classes
    
    # Download CIFAR-100
    print("Downloading/loading CIFAR-100...")
    cifar100 = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    cifar100_classes = cifar100.classes
    
    print(f"✅ CIFAR-10: {len(cifar10)} test images, {len(cifar10_classes)} classes")
    print(f"✅ CIFAR-100: {len(cifar100)} test images, {len(cifar100_classes)} classes")
    
    return cifar10, cifar100, cifar10_classes, cifar100_classes

def evaluate_zero_shot(model_name, model, processor, dataloader, classnames, dataset_name):
    """Evaluate a single model for zero-shot classification"""
    print(f"\n🔍 Evaluating {model_name} on {dataset_name}...")
    
    # Create prompts with template - more specific for CIFAR
    if "cifar" in dataset_name.lower():
        prompts = [f"a photo of a {name}, a type of object" for name in classnames]
    else:
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
    for images, labels in tqdm(dataloader, desc=f"Processing {dataset_name}"):
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
                values, indices = similarity.topk(min(5, len(classnames)), dim=1)
                
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
                if total % 1000 == 0 or total == len(dataloader.dataset):
                    interim_top1 = correct_top1 / total
                    interim_top5 = correct_top5 / total
                    print(f"\nAfter {total} examples - {model_name} Top-1: {interim_top1:.4f}, Top-5: {interim_top5:.4f}")
                
            except Exception as e:
                print(f"Error processing batch: {e}")
    
    # Calculate accuracy
    top1 = correct_top1 / total if total > 0 else 0
    top5 = correct_top5 / total if total > 0 else 0
    
    print(f"{model_name} {dataset_name} Final Results - Top-1: {top1:.4f}, Top-5: {top5:.4f}")
    
    return {"top1": top1, "top5": top5}

def main():
    try:
        print("===== CIFAR Zero-Shot Evaluation =====")
        
        # Download CIFAR datasets
        cifar10, cifar100, cifar10_classes, cifar100_classes = download_cifar_datasets(data_dir)
        
        # Create dataloaders with larger batch size for efficiency
        cifar10_loader = torch.utils.data.DataLoader(
            cifar10, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
            
        cifar100_loader = torch.utils.data.DataLoader(
            cifar100, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        
        # Load models
        print("Loading models...")
        base_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # Evaluate base CLIP model on CIFAR-10
        print("\n==== Base CLIP Model ====")
        base_cifar10_results = evaluate_zero_shot(
            "base", base_model, clip_processor, cifar10_loader, cifar10_classes, "CIFAR-10")
        base_cifar100_results = evaluate_zero_shot(
            "base", base_model, clip_processor, cifar100_loader, cifar100_classes, "CIFAR-100")
        
        # Load and evaluate your custom model
        try:
            print("\n==== Custom Model ====")
            print("Loading custom model...")
            custom_model = CLIPImageDistillation.load_from_checkpoint(
                checkpoint_path,
                map_location=device,
                clip_model=base_model,
                clip_preprocess=clip_processor,
                strict=False
            ).to(device).eval()
            
            # Evaluate on CIFAR-10
            custom_cifar10_results = evaluate_zero_shot(
                "custom", custom_model, clip_processor, cifar10_loader, cifar10_classes, "CIFAR-10")
            
            # Evaluate on CIFAR-100
            custom_cifar100_results = evaluate_zero_shot(
                "custom", custom_model, clip_processor, cifar100_loader, cifar100_classes, "CIFAR-100")
            
        except Exception as e:
            print(f"Error loading custom model: {e}")
            custom_cifar10_results = {"top1": 0, "top5": 0}
            custom_cifar100_results = {"top1": 0, "top5": 0}
        
        # Display results
        print("\n📊 Zero-Shot CIFAR Results")
        print("=" * 70)
        print(f"{'Model':<15} {'Dataset':<10} {'Top-1 Acc':<15} {'Top-5 Acc':<15} {'Rel. Change':<15}")
        print("-" * 70)
        print(f"Base CLIP       CIFAR-10   {base_cifar10_results['top1']:.4f}         {base_cifar10_results['top5']:.4f}         -")
        print(f"Custom Model    CIFAR-10   {custom_cifar10_results['top1']:.4f}         {custom_cifar10_results['top5']:.4f}         {(custom_cifar10_results['top1']-base_cifar10_results['top1'])/base_cifar10_results['top1']*100:+.2f}%")
        print(f"Base CLIP       CIFAR-100  {base_cifar100_results['top1']:.4f}         {base_cifar100_results['top5']:.4f}         -")
        print(f"Custom Model    CIFAR-100  {custom_cifar100_results['top1']:.4f}         {custom_cifar100_results['top5']:.4f}         {(custom_cifar100_results['top1']-base_cifar100_results['top1'])/base_cifar100_results['top1']*100:+.2f}%")
        
        # Save results
        with open("cifar_zero_shot_results.txt", "w") as f:
            f.write("Zero-Shot CIFAR Results\n")
            f.write("=" * 70 + "\n")
            f.write(f"CIFAR-10:\n")
            f.write(f"Base CLIP Top-1: {base_cifar10_results['top1']:.4f}, Top-5: {base_cifar10_results['top5']:.4f}\n")
            f.write(f"Custom Model Top-1: {custom_cifar10_results['top1']:.4f}, Top-5: {custom_cifar10_results['top5']:.4f}\n")
            f.write(f"Relative Change: {(custom_cifar10_results['top1']-base_cifar10_results['top1'])/base_cifar10_results['top1']*100:+.2f}%\n\n")
            
            f.write(f"CIFAR-100:\n")
            f.write(f"Base CLIP Top-1: {base_cifar100_results['top1']:.4f}, Top-5: {base_cifar100_results['top5']:.4f}\n")
            f.write(f"Custom Model Top-1: {custom_cifar100_results['top1']:.4f}, Top-5: {custom_cifar100_results['top5']:.4f}\n")
            f.write(f"Relative Change: {(custom_cifar100_results['top1']-base_cifar100_results['top1'])/base_cifar100_results['top1']*100:+.2f}%\n")
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()