import os
import json
import torch
import faiss
import numpy as np
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

#MAY NEED PYTORCH VERSION 1.23.5

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv5 model
yolo_model = YOLO("yolov5s.pt")  # Small YOLO model

# Load CLIP model from Hugging Face transformers
model_id = "openai/clip-vit-base-patch32"  # This is equivalent to ViT-B/32
clip_model = CLIPModel.from_pretrained(model_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_id)

# Initialize FAISS Index
embedding_dim = 512  # CLIP ViT-B/32 output dim
index = faiss.IndexFlatIP(embedding_dim)

# Data paths
image_dir = "C:/Users/Chuck/Desktop/Coding/CodingProjects/MultiModalBPR/data/small_train2014"
embedding_json_path = "clip_embeddings_coco.json"
faiss_index_path = "faiss_clip_index_coco.idx"

# Store embeddings in a dictionary
embedding_dict = {}

def get_clip_embedding(image):
    # Process the image using the CLIP processor
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    
    # Get image features
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        embedding = image_features.cpu().numpy()
    
    # Normalize embedding for cosine similarity
    normalized_embedding = embedding / np.linalg.norm(embedding)
    return normalized_embedding

# Process images
for img_name in tqdm(os.listdir(image_dir), desc="Processing images"):
    img_path = os.path.join(image_dir, img_name)

    # Load image
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {img_path}: {e}")
        continue

    # Run YOLO on the image
    results = yolo_model(img_path)

    # Process detected objects
    for i, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            try:
                # Crop patch
                patch = image.crop((x1, y1, x2, y2))
                
                # Get CLIP embedding
                embedding = get_clip_embedding(patch).astype(np.float32)

                # Store in FAISS index
                index.add(embedding)

                # Save in dictionary
                patch_id = f"{img_name}_patch{j}"
                embedding_dict[patch_id] = embedding.tolist()
                
                # Store patch position for reference
                position = [x1/image.width, y1/image.height, x2/image.width, y2/image.height]
                if patch_id in embedding_dict:
                    embedding_dict[patch_id] = {
                        "embedding": embedding.tolist(),
                        "position": position
                    }
            except Exception as e:
                print(f"Error processing patch from {img_path}: {e}")
                continue

# Save FAISS index
faiss.write_index(index, faiss_index_path)
print(f"FAISS index saved to {faiss_index_path} with {index.ntotal} vectors")

# Save embeddings to JSON
with open(embedding_json_path, "w") as f:
    json.dump(embedding_dict, f)
print(f"Embeddings saved to {embedding_json_path} with {len(embedding_dict)} entries")

# Optional: Print some statistics
print(f"Total processed patches: {index.ntotal}")