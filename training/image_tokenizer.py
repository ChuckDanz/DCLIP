# image_tokenizer.py

from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw
import contextlib
import os
import torchvision.transforms as T
import faiss
import json
import numpy as np
#import clip  # Make sure this is included
from image_projection_module import ImageProjectionModule



#ENSURE THAT TEXT TOKENIZER IS ON GPU                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
class CLIPPatchTokenizer:
    def __init__(self, clip_model_id="openai/clip-vit-base-patch16"):
        self.clip_preprocess = CLIPProcessor.from_pretrained(clip_model_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO("./yolov8x.pt").to(self.device)  # Load YOLOv8 model
        # Load the CLIP model for encoding patches (if not already done elsewhere)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(self.device)


        self.patch_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
            
            ])
        
        self.full_res_transform = T.ToTensor()

    def get_weighted_bounding_boxes(self, image_path):
        """
        Gets bounding boxes with weights (e.g., using detection confidence).

        Args:
            image_path (str): Path to the input image.
        
        Returns:
            list: List of tuples ( (x1, y1, x2, y2), confidence ).
        """
        with torch.no_grad():
            # Suppress YOLO logging
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    results = self.model(image_path)

        result = results[0]
        weighted_boxes = []
        # result.boxes.xyxy, result.boxes.conf should be available
        for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            weighted_boxes.append(((x1, y1, x2, y2), float(conf)))
        return weighted_boxes
    
    def get_weighted_bounding_boxes_batch(self, image_paths, mini_batch_size=16):
        """
        Gets bounding boxes with weights for a batch of images in smaller sub-batches.
        
        Args:
            image_paths (list): List of image file paths.
            mini_batch_size (int): The size of each mini-batch.
            
        Returns:
            dict: Mapping from image_path to list of tuples ((x1, y1, x2, y2), confidence).
        """
        boxes_dict = {}
        for i in range(0, len(image_paths), mini_batch_size):
            mini_batch = image_paths[i:i + mini_batch_size]
            with torch.no_grad():
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        results = self.model(mini_batch)
            for image_path, result in zip(mini_batch, results):
                weighted_boxes = []
                for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box)
                    weighted_boxes.append(((x1, y1, x2, y2), float(conf)))
                boxes_dict[image_path] = weighted_boxes
        return boxes_dict

    def encode_weighted_bounding_boxes(self, image, weighted_boxes, full_resolution=False):
        """
        Extracts patches from an image for all weighted boxes at once,
        then runs a single batch forward pass through the CLIP model.
        Returns a list of (clip_embedding, confidence) tuples.
        
        Args:
            image (PIL.Image): The input image.
            weighted_boxes (list): List of tuples ((x1, y1, x2, y2), confidence).
            full_resolution (bool): If True, do not resize patch to 224x224.
                                      Defaults to False.
        """
        patch_tensors = []
        confidences = []
        for box, conf in weighted_boxes:
            x1, y1, x2, y2 = box
            # Crop the patch from the image.
            patch = image.crop((x1, y1, x2, y2))
            # Use different transforms depending on full_resolution flag:
            if full_resolution:
                patch_tensor = self.full_res_transform(patch)
            else:
                patch_tensor = self.patch_transform(patch)
            patch_tensors.append(patch_tensor)
            confidences.append(conf)

        if patch_tensors:
            # Stack all patches into a single tensor [B, C, H, W] and send to GPU.
            batch = torch.stack(patch_tensors).to(self.device)
            # Get the dtype of the CLIP model weights.
            target_dtype = next(self.clip_model.parameters()).dtype
            # Cast the batch to that dtype.
            inputs = {"pixel_values": batch.to(dtype=target_dtype)}
            with torch.no_grad():
                patch_embs = self.clip_model.get_image_features(**inputs)
            embeddings = [(emb, conf) for emb, conf in zip(patch_embs, confidences)]
        else:
            embeddings = []
        return embeddings

    def encode_bounding_boxes_with_context(self, image, weighted_boxes):
        """
        For each bounding box, produces patch and context embeddings using batching.
        The context image is computed by blacking out the cropped region.
        Returns a list of (patch_embedding, context_embedding, confidence) tuples.
        """
        patch_list = []
        context_list = []
        confidences = []
        for box, conf in weighted_boxes:
            x1, y1, x2, y2 = box
            # (1) Crop the patch
            patch = image.crop((x1, y1, x2, y2))
            patch_tensor = self.patch_transform(patch)  # [C, 224, 224]
            patch_list.append(patch_tensor)
            # (2) Create context image by blacking out the box
            context_img = image.copy()
            draw = ImageDraw.Draw(context_img)
            draw.rectangle([x1, y1, x2, y2], fill="black")
            context_tensor = self.patch_transform(context_img)  # [C, 224, 224]
            context_list.append(context_tensor)
            confidences.append(conf)

        if patch_list:
            # Stack the patch and context tensors and send to GPU
            batch_patch = torch.stack(patch_list).to(self.device)
            batch_context = torch.stack(context_list).to(self.device)
            # Get target dtype from the CLIP model weights
            target_dtype = next(self.clip_model.parameters()).dtype
            with torch.no_grad():
                patch_embs = self.clip_model.get_image_features(pixel_values=batch_patch.to(dtype=target_dtype))  # [B, 512]
                context_embs = self.clip_model.get_image_features(pixel_values=batch_context.to(dtype=target_dtype))  # [B, 512]
            context_patch_pairs = [
                (patch_emb, context_emb, conf)
                for patch_emb, context_emb, conf in zip(patch_embs, context_embs, confidences)
            ]
        else:
            context_patch_pairs = []
        return context_patch_pairs

class ImageTokenizer:
    def __init__(
        self,
        clip_model_id="openai/clip-vit-base-patch16", # Changed to use HF model ID #openai/clip-vit-large-patch14
        projection_model_path=None,  
        faiss_index_path=None,      
        embeddings_json_path=None,   
        similarity_threshold=0.85,  
        device=None                  
    ):
        # Setup device
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ImageTokenizer using device: {self.device}")
        
        # Load CLIP model using HuggingFace
        print(f"Loading CLIP model: {clip_model_id}")
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(self.device)
        self.preprocess = CLIPProcessor.from_pretrained(clip_model_id)
        
        # Set configuration
        self.similarity_threshold = similarity_threshold
        self.use_knn = False
        
        # Load Projection Module (if provided)
        self.projection_module = None
        if projection_model_path and os.path.exists(projection_model_path):
            print(f"Loading projection module from: {projection_model_path}")
            
            # Load the checkpoint which contains more than just the model weights
            checkpoint = torch.load(projection_model_path, map_location=self.device)
            
            # Initialize the model
            self.projection_module = ImageProjectionModule(clip_dim=512, hidden_dim=1024)

            # Check if it's a full training checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Extract just the model weights
                print("Loading from full checkpoint (extracting model_state_dict)")
                self.projection_module.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try loading directly (in case it's just a state_dict)
                print("Loading state dictionary directly")
                self.projection_module.load_state_dict(checkpoint)
                
            self.projection_module.to(self.device)
            self.projection_module.eval()
        else:
            print("No projection module provided or file not found.")
        
        # Load FAISS resources (if provided)
        self.faiss_index = None
        self.patch_ids = None
        self.embedding_dict = None
        
        if faiss_index_path and embeddings_json_path and os.path.exists(faiss_index_path) and os.path.exists(embeddings_json_path):
            print(f"Loading FAISS index from: {faiss_index_path}")
            self.faiss_index = faiss.read_index(faiss_index_path)
            
            print(f"Loading embeddings from: {embeddings_json_path}")
            with open(embeddings_json_path, 'r') as f:
                self.embedding_dict = json.load(f)
            self.patch_ids = list(self.embedding_dict.keys())
            self.use_knn = True
            print(f"FAISS index loaded with {len(self.patch_ids)} patch embeddings")
        else:
            print("No FAISS index provided or files not found. KNN search will be disabled.")

    def get_clip_embedding(self, image):
        """Compute normalized CLIP embedding for an image using HF CLIP"""
        if isinstance(image, str):
            # If image is a file path, open the image
            image = Image.open(image).convert("RGB")
            
        # Preprocess and get embedding using HF's CLIP
        inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs).cpu().numpy()
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype(np.float32)

    def get_patch_embedding(self, image_or_path, position=None):
        """
        Get patch embedding using hybrid approach:
        1. Check KNN for similar patches (if FAISS index loaded)
        2. Fall back to projection module if similarity below threshold
        """
         # Get CLIP embedding for the image
        clip_embedding = self.get_clip_embedding(image_or_path)
        
        # Use KNN if available and configured
        if self.use_knn and self.faiss_index is not None:
            # Search for similar patches - get 3 nearest neighbors for better analysis
            k = 3
            distances, indices = self.faiss_index.search(clip_embedding, k)
            
            # Get the closest match
            if len(distances[0]) > 0:
                max_similarity = float(distances[0][0])
                
                # CRITICAL FIX: Debug actual values from FAISS
                print(f"DEBUG: Raw FAISS scores: {distances[0]}")
                
                # CRITICAL FIX: Apply correct comparison - add a scale factor
                # FAISS inner product scores may need scaling depending on your vectors
                adjusted_score = max_similarity * 1.0  # Adjust this factor if needed
                
                print(f"KNN similarity: {adjusted_score:.4f}, threshold: {self.similarity_threshold:.4f}")
            
            # Check if similarity meets threshold
            if adjusted_score >= self.similarity_threshold:
                closest_patch_id = self.patch_ids[indices[0][0]]
                retrieved_embedding = np.array(self.embedding_dict[closest_patch_id])
                print(f"✓ Using KNN, patch_id={closest_patch_id}")
                return retrieved_embedding, "knn", max_similarity
            else:
                print(f"✗ KNN similarity {adjusted_score:.4f} below threshold, using projection")
        else:
            print("No KNN matches found!")
        
        # Fall back to projection module if available
        if self.projection_module is not None:
            # Convert to torch tensor
            clip_embedding_tensor = torch.from_numpy(clip_embedding).to(self.device)
            
            # Create position tensor if provided
            if position is not None:
                position_tensor = torch.tensor([position]).to(self.device)
            else:
                # Default position if not provided
                position_tensor = torch.zeros(1, 4).to(self.device)
            
            # Get projection
            with torch.no_grad():
                projected_embedding = self.projection_module(
                    clip_embedding_tensor, 
                    position_tensor
                ).cpu().numpy()
            
            # Normalize the embedding
            projected_embedding = projected_embedding / np.linalg.norm(projected_embedding)
            print("Using projection module")
            
            return projected_embedding[0], "projection", 0.0
        
        # If neither KNN nor projection available, return CLIP embedding directly
        print("Using raw CLIP embedding")
        return clip_embedding[0], "clip", 0.0
        
        
        
    
    def tokenize(self, image_or_path, position=None):
        """
        Main tokenization function that converts an image to a token embedding.
        This is the primary entry point for the tokenizer.
        """
        embedding, source, similarity = self.get_patch_embedding(image_or_path, position)
        return embedding


# Add this class after your existing ImageTokenizer class

class TokenizerWithKNN:
    """
    Unified image tokenizer that combines both your CLIPPatchTokenizer (for object detection and patch extraction)
    and the ImageTokenizer (for KNN retrieval and projection).
    
    This creates a complete image-side counterpart to your text_tokenizer.
    """
    def __init__(
        self,
        clip_model_id="openai/clip-vit-base-patch16",
        projection_model_path=None,
        faiss_index_path=None,
        embeddings_json_path=None,
        similarity_threshold=0.85,
        yolo_model_path="./yolov8x.pt",  # Path to YOLO model
        use_detection=True,  # Whether to use object detection
        device=None
    ):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize the detection-based tokenizer
        if use_detection:
            self.patch_tokenizer = CLIPPatchTokenizer(clip_model_id)
            print("Initialized patch extraction with YOLOv8")
        else:
            self.patch_tokenizer = None
        
        # Initialize the KNN/projection tokenizer
        self.knn_tokenizer = ImageTokenizer(
            clip_model_id=clip_model_id,
            projection_model_path=projection_model_path,
            faiss_index_path=faiss_index_path,
            embeddings_json_path=embeddings_json_path,
            similarity_threshold=similarity_threshold,
            device=self.device
        )
    
    def tokenize_image(self, image_path, use_patches=True, top_k_detections=3):
        """
        Main entry point for image tokenization
        
        Args:
            image_path: Path to the image file
            use_patches: If True, use object detection to extract patches. Otherwise, use whole image.
            top_k_detections: Number of top detections to use (if use_patches=True)
            
        Returns:
            List of embeddings (if use_patches=True) or single embedding (if use_patches=False)
        """
        if use_patches and self.patch_tokenizer:
            # Get bounding boxes with confidence scores
            weighted_boxes = self.patch_tokenizer.get_weighted_bounding_boxes(image_path)
            
            # Sort by confidence and take top k
            weighted_boxes.sort(key=lambda x: x[1], reverse=True)
            weighted_boxes = weighted_boxes[:top_k_detections]
            
            # Load the image
            image = Image.open(image_path).convert("RGB")
            
            # Process each detection
            results = []
            for (x1, y1, x2, y2), confidence in weighted_boxes:
                # Convert to normalized coords for position
                w, h = image.size
                position = [x1/w, y1/h, x2/w, y2/h]
                
                # Extract patch
                patch = image.crop((x1, y1, x2, y2))
                
                # Run through KNN/projection pipeline
                embedding, source, similarity = self.knn_tokenizer.get_patch_embedding(patch, position)
                
                # Store result with metadata
                results.append({
                    'embedding': embedding,
                    'confidence': confidence,
                    'position': position,
                    'source': source,
                    'similarity': similarity
                })
            
            return results
        else:
            # Process the whole image
            embedding = self.knn_tokenizer.tokenize(image_path)
            return embedding
    
    def knn_or_projection(self, image_path, position=None):
        """
        Direct access to KNN/projection decision for a single image/patch
        
        Args:
            image_path: Path to the image
            position: Optional position [x1/W, y1/H, x2/W, y2/H]
            
        Returns:
            embedding, source, similarity
        """
        return self.knn_tokenizer.get_patch_embedding(image_path, position)

    def batch_tokenize(self, image_paths, use_patches=True, top_k_detections=3):
        """
        Process multiple images efficiently in batch mode
        
        Args:
            image_paths (list): List of paths to image files
            use_patches (bool): Whether to use object detection
            top_k_detections (int): Number of top detections per image
            
        Returns:
            dict: Mapping from image path to tokenization results
        """
        results = {}
        
        if use_patches and self.patch_tokenizer:
            # Get all bounding boxes in mini-batches for efficiency
            boxes_dict = self.patch_tokenizer.get_weighted_bounding_boxes_batch(image_paths)
            
            for image_path, weighted_boxes in boxes_dict.items():
                # Sort by confidence and take top k
                weighted_boxes.sort(key=lambda x: x[1], reverse=True)
                weighted_boxes = weighted_boxes[:top_k_detections]
                
                # Process each image
                image = Image.open(image_path).convert("RGB")
                image_results = []
                
                for (x1, y1, x2, y2), confidence in weighted_boxes:
                    # Convert to normalized coords
                    w, h = image.size
                    position = [x1/w, y1/h, x2/w, y2/h]
                    
                    # Extract patch
                    patch = image.crop((x1, y1, x2, y2))
                    
                    # Run through KNN/projection pipeline
                    embedding, source, similarity = self.knn_tokenizer.get_patch_embedding(patch, position)
                    
                    # Store result with metadata
                    image_results.append({
                        'embedding': embedding,
                        'confidence': confidence,
                        'position': position,
                        'bbox': (x1, y1, x2, y2),
                        'source': source,
                        'similarity': similarity
                    })
                
                results[image_path] = image_results
        else:
            # Process whole images
            for image_path in image_paths:
                embedding = self.knn_tokenizer.tokenize(image_path)
                results[image_path] = embedding
        
        return results

    def evaluate_threshold(self, image_path, thresholds=[0.6, 0.7, 0.8, 0.85, 0.9, 0.95]):
        """
        Evaluate different similarity thresholds to find the optimal value
        
        Args:
            image_path: Path to test image
            thresholds: List of thresholds to evaluate
        
        Returns:
            dict: Mapping from threshold to source method used
        """
        results = {}
        original_threshold = self.knn_tokenizer.similarity_threshold
        
        # Get weighted boxes
        if self.patch_tokenizer:
            weighted_boxes = self.patch_tokenizer.get_weighted_bounding_boxes(image_path)
            # Sort by confidence and take top 1
            weighted_boxes.sort(key=lambda x: x[1], reverse=True)
            box, _ = weighted_boxes[0] if weighted_boxes else ((0, 0, 100, 100), 0.0)
            
            # Load image and extract patch
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            x1, y1, x2, y2 = box
            position = [x1/w, y1/h, x2/w, y2/h]
            patch = image.crop((x1, y1, x2, y2))
            
            # Test each threshold
            for threshold in thresholds:
                self.knn_tokenizer.similarity_threshold = threshold
                _, source, similarity = self.knn_tokenizer.get_patch_embedding(patch, position)
                results[threshold] = {
                    'source': source,
                    'similarity': similarity if source == 'knn' else 0.0
                }
        
        # Restore original threshold
        self.knn_tokenizer.similarity_threshold = original_threshold
        return results

# Example usage
if __name__ == "__main__":
    # Paths to resources
    model_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(model_dir)
    
    projection_path = os.path.join(parent_dir, "train/output/best_model.pth")
    dataset_dir = os.path.join(parent_dir, "ImageProjectionModuleDev/datasets/masked_image_ds")
    faiss_dir = os.path.join(dataset_dir, "faiss")
    faiss_index_path = os.path.join(faiss_dir, "faiss_clip_index.idx")
    embeddings_json_path = os.path.join(faiss_dir, "clip_embeddings.json")
    
    # Initialize the tokenizer
    tokenizer = ImageTokenizer(
        projection_model_path=projection_path,
        faiss_index_path=faiss_index_path,
        embeddings_json_path=embeddings_json_path,
        similarity_threshold=0.85
    )
    
    # Test with a sample image
    sample_image_path = os.path.join(dataset_dir, "patch/patch_1.png")
    if os.path.exists(sample_image_path):
        embedding = tokenizer.tokenize(sample_image_path)
        print(f"Embedding shape: {embedding.shape}")
    else:
        print(f"Sample image not found: {sample_image_path}")
        print("Please provide a valid image path to test.")


