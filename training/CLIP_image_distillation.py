import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from image_tokenizer import CLIPPatchTokenizer
from PIL import Image
import json
from transformers import get_linear_schedule_with_warmup
import argparse
import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import pickle
import torchvision
from torchvision import transforms

from patch_text_aggregation import PatchTextAggregation

from image_tokenizer import CLIPPatchTokenizer

import random

import math

import copy

tokenizer = CLIPPatchTokenizer()

# learn about opitimzers and data workers

#MAKE SURE MULTI PHASE TRAINING WORKS
#START TRAINING THE TEXT ENCODER ON PRE ATTENDED TEXT EMBEDDINGS, figure out some aggregation system
#so chunk the text up and aggregate pre attended text embeddings, then distill with the student one


#for full assembly precompute if a patch is unknown and store in cache, then run projection module

#need to test the model and make sure full resolution is working


def load_or_compute_yolo(image_path, clip_preprocess, cache_dir="./cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(image_path) + ".pkl")
    
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                weighted_boxes = pickle.load(f)
                return weighted_boxes
    except (pickle.UnpicklingError, EOFError, ImportError) as e:
        print(f"Warning: Corrupt cache file detected for {os.path.basename(image_path)}, regenerating...")
        # If file is corrupt, remove it so we regenerate
        if os.path.exists(cache_file):
            os.remove(cache_file)
    
    # If we reach here, we need to compute the boxes
    from image_tokenizer import CLIPPatchTokenizer
    tokenizer = CLIPPatchTokenizer()
    weighted_boxes = tokenizer.get_weighted_bounding_boxes(image_path)
    
    # Use atomic write pattern to prevent corruption
    temp_file = cache_file + ".tmp"
    with open(temp_file, "wb") as f:
        pickle.dump(weighted_boxes, f)
    
    # Atomic rename to final location
    if os.path.exists(temp_file):  # Check that write succeeded
        if os.path.exists(cache_file):
            os.remove(cache_file)  # Remove old file if it exists
        os.rename(temp_file, cache_file)
    
    return weighted_boxes


class MultiModalDataset(Dataset):
    def __init__(self, json_file, clip_preprocess, cache_dir="./cache", use_batch_cache=True, cache_filename="train_precache.pkl"):
        """
        Initialize the dataset with memory-efficient caching to prevent OOM errors
        """
        # Force garbage collection at start
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load dataset JSON with better error handling
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Error loading dataset JSON: {e}")
            # Default to empty list if JSON can't be loaded
            self.data = []
        
        # Initialize basic properties
        self.preprocess = clip_preprocess
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.use_batch_cache = use_batch_cache
        self.cache_filename = cache_filename

        # Set up memory efficient numpy
        import numpy as np
        np.zeros(1).astype(np.float32)  # Force NumPy to initialize with float32
        
        # Initialize empty cache dictionary
        self.cached_yolo = {}
        
        # Use memory-mapped approach for cache
        if self.use_batch_cache:
            precache_file = os.path.join(self.cache_dir, self.cache_filename)
            mmap_file = os.path.join(self.cache_dir, f"{os.path.splitext(self.cache_filename)[0]}_mmap.db")
            keys_file = os.path.join(self.cache_dir, f"{os.path.splitext(self.cache_filename)[0]}_keys.pkl")
            
            # Strategy 1: Load existing cache if available
            if os.path.exists(precache_file):
                print(f"Loading cache from: {precache_file}")
                try:
                    # Memory-efficient approach - load cache in chunks instead of all at once
                    cache_size = os.path.getsize(precache_file)
                    print(f"Cache size: {cache_size / (1024*1024):.2f} MB")
                    
                    # If cache is small enough (< 1GB), load directly
                    if cache_size < 1024*1024*1024:  # Less than 1GB
                        with open(precache_file, 'rb') as f:
                            self.cached_yolo = pickle.load(f)
                        print(f"Loaded regular cache with {len(self.cached_yolo)} entries")
                    else:
                        # For large caches, use a dictionary-like object that loads entries on demand
                        import dbm
                        
                        # Check if we already have a memory-mapped version
                        if os.path.exists(mmap_file + ".dat"):
                            print(f"Using existing memory-mapped cache")
                            self.mmap_db = dbm.open(mmap_file, 'r')
                            
                            # Load keys separately - much smaller memory footprint
                            if os.path.exists(keys_file):
                                with open(keys_file, 'rb') as f:
                                    self.cache_keys = pickle.load(f)
                                print(f"Loaded {len(self.cache_keys)} keys from {keys_file}")
                            else:
                                # Generate keys from the DB if needed
                                self.cache_keys = list(self.mmap_db.keys())
                                
                            # Create a proxy dictionary that loads from disk on demand
                            class DiskCache(dict):
                                def __init__(self, db, keys):
                                    self.db = db
                                    self.keys_list = keys
                                    
                                def __getitem__(self, key):
                                    if isinstance(key, str) and key.encode('utf-8') in self.db:
                                        return pickle.loads(self.db[key.encode('utf-8')])
                                    return None
                                    
                                def __contains__(self, key):
                                    return isinstance(key, str) and key.encode('utf-8') in self.db
                                    
                                def get(self, key, default=None):
                                    try:
                                        if key in self:
                                            return self[key]
                                    except:
                                        pass
                                    return default
                                    
                                def __len__(self):
                                    return len(self.keys_list)
                            
                            # Use the proxy dictionary instead of loading everything
                            self.cached_yolo = DiskCache(self.mmap_db, self.cache_keys)
                            print(f"Using disk cache with {len(self.cached_yolo)} entries")
                        else:
                            # Convert regular cache to memory-mapped if too large
                            print("Converting large cache to memory-mapped format (this will happen once)...")
                            try:
                                # Create new memory-mapped DB
                                self.mmap_db = dbm.open(mmap_file, 'c')
                                
                                # Process the pickle file in chunks to avoid memory issues
                                import io
                                
                                # Extract and store keys first (much smaller)
                                key_list = []
                                
                                # Open file and read in chunks
                                with open(precache_file, 'rb') as f:
                                    # Load pickle dictionary in a memory-efficient way using a streaming approach
                                    #import pickle5 as pickle  # Better for large files if available
                                    
                                    # Try streaming unpickler approach
                                    try:
                                        cache_data = pickle.load(f)
                                        print(f"Processing {len(cache_data)} entries")
                                        
                                        # Process in manageable chunks
                                        batch_size = 1000
                                        keys = list(cache_data.keys())
                                        key_list = keys  # Save keys for later
                                        
                                        for i in range(0, len(keys), batch_size):
                                            batch_keys = keys[i:i+batch_size]
                                            for key in batch_keys:
                                                self.mmap_db[key.encode('utf-8')] = pickle.dumps(cache_data[key])
                                            
                                            if i % 10000 == 0:
                                                print(f"Processed {i} of {len(keys)} entries...")
                                                # Force garbage collection
                                                del batch_keys
                                                gc.collect()
                                        
                                        # Save keys separately
                                        with open(keys_file, 'wb') as kf:
                                            pickle.dump(key_list, kf)
                                        
                                        print(f"Converted cache to memory-mapped format")
                                        
                                        # Create the proxy dictionary
                                        class DiskCache(dict):
                                            def __init__(self, db, keys):
                                                self.db = db
                                                self.keys_list = keys
                                                
                                            def __getitem__(self, key):
                                                if isinstance(key, str) and key.encode('utf-8') in self.db:
                                                    return pickle.loads(self.db[key.encode('utf-8')])
                                                return None
                                                
                                            def __contains__(self, key):
                                                return isinstance(key, str) and key.encode('utf-8') in self.db
                                                
                                            def get(self, key, default=None):
                                                try:
                                                    if key in self:
                                                        return self[key]
                                                except:
                                                    pass
                                                return default
                                                
                                            def __len__(self):
                                                return len(self.keys_list)
                                        
                                        # Use the proxy dictionary
                                        self.cached_yolo = DiskCache(self.mmap_db, key_list)
                                        print(f"Using disk cache with {len(self.cached_yolo)} entries")
                                        
                                        # Clear the original cache from memory
                                        del cache_data
                                        gc.collect()
                                        
                                    except Exception as inner_e:
                                        print(f"Error in streaming approach: {inner_e}")
                                        # Fall back to empty cache
                                        self.cached_yolo = {}
                            
                            except Exception as outer_e:
                                print(f"Error converting cache: {outer_e}")
                                # Fall back to empty cache if conversion fails
                                self.cached_yolo = {}
                
                except (pickle.UnpicklingError, EOFError, ImportError, MemoryError) as e:
                    print(f"Warning: Cache file error ({e}), regenerating...")
                    self.cached_yolo = {}  # Reset to empty
            
            # Strategy 2: If no cache exists or loading failed, generate it with memory efficiency
            if not self.cached_yolo:
                print("No usable cache found, generating new cache...")
                # Extract image_paths - handle both old and new format with memory efficiency
                image_paths = []
                for item in self.data:
                    if "image_path" in item:
                        image_paths.append(item["image_path"])
                
                # Process images in smaller batches to avoid memory issues
                batch_size = 32  # Smaller batch size
                tokenizer = CLIPPatchTokenizer()
                
                for i in range(0, len(image_paths), batch_size):
                    print(f"Processing batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1}")
                    batch_paths = image_paths[i:i+batch_size]
                    batch_results = tokenizer.get_weighted_bounding_boxes_batch(batch_paths)
                    
                    # Update cache
                    for path, boxes in batch_results.items():
                        self.cached_yolo[path] = boxes
                    
                    # Force garbage collection after each batch
                    gc.collect()
                
                print("Saving cache...")
                # Save in a memory-efficient way
                try:
                    # Use atomic write pattern to prevent corruption
                    temp_file = precache_file + ".tmp"
                    with open(temp_file, 'wb') as f:
                        pickle.dump(self.cached_yolo, f, protocol=4)  # Use protocol 4 for better efficiency
                        
                    # Atomic rename to final location
                    if os.path.exists(temp_file):  # Check that write succeeded
                        if os.path.exists(precache_file):
                            os.remove(precache_file)  # Remove old file if it exists
                        os.rename(temp_file, precache_file)
                        print(f"Cache saved to {precache_file}")
                except Exception as e:
                    print(f"Error saving cache: {e}")

        # Final memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            # Try to get a valid item, skip problematic ones if needed
        max_retries = 3
        current_idx = idx
        
        for retry in range(max_retries):
            try:
                item = self.data[current_idx]
                image_path = item.get('image_path', "")
                
                # Handle both old and new format for captions
                if "captions" in item:
                    # New format - multiple captions, randomly select one
                    captions = item.get("captions", [])
                    caption = random.choice(captions) if captions else ""
                else:
                    # Old format - single caption
                    caption = item.get('caption', "")
                
                # Ensure image is loaded as RGB - critical fix for the normalization error
                try:
                    with Image.open(image_path) as img:
                        # Explicitly convert to RGB mode (even if already RGB)
                        # The .copy() ensures we don't have any reference issues
                        image = img.convert("RGB").copy()
                        
                        # Quick sanity check - verify it's truly RGB
                        if image.mode != "RGB":
                            raise ValueError(f"Failed to convert image to RGB mode: {image_path}")
                    
                    # Preprocess image for the student model - with error handling
                    processed = self.preprocess(images=image, text="", return_tensors="pt")
                    pixel_values = processed["pixel_values"].squeeze(0)
                    
                    # If batch caching is enabled, try to load cached YOLO detections
                    if self.use_batch_cache:
                        weighted_boxes = self.cached_yolo.get(image_path, None)
                        if weighted_boxes is None:
                            # Compute YOLO detections live and update in-memory cache only
                            weighted_boxes = load_or_compute_yolo(image_path, self.preprocess, cache_dir=self.cache_dir)
                            self.cached_yolo[image_path] = weighted_boxes
                            
                            # Only try to update disk cache from main process or if single-process
                            worker_info = torch.utils.data.get_worker_info()
                            is_main_process = worker_info is None or worker_info.id == 0
                            
                            if is_main_process:
                                try:
                                    precache_file = os.path.join(self.cache_dir, self.cache_filename)
                                    temp_file = precache_file + ".tmp"
                                    with open(temp_file, 'wb') as f:
                                        pickle.dump(self.cached_yolo, f)
                                    if os.path.exists(temp_file):
                                        if os.path.exists(precache_file):
                                            try:
                                                os.remove(precache_file)
                                            except PermissionError:
                                                print(f"Warning: Could not update cache - file in use")
                                                # Skip updating if file is locked
                                                if os.path.exists(temp_file):
                                                    os.remove(temp_file)
                                        try:
                                            os.rename(temp_file, precache_file)
                                        except PermissionError:
                                            print(f"Warning: Could not rename temp cache file - in use")
                                            if os.path.exists(temp_file):
                                                os.remove(temp_file)
                                except Exception as e:
                                    print(f"Cache update error: {e}")
                    else:
                        # Otherwise fallback to per-image caching
                        weighted_boxes = load_or_compute_yolo(image_path, self.preprocess, cache_dir=self.cache_dir)
                    
                    # Successfully processed this item
                    return pixel_values, caption, image_path, weighted_boxes
                    
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    # Try again with next retry
                    raise
                    
            except Exception as e:
                # Problem with this image, move to next one if we have retries left
                if retry < max_retries - 1:
                    current_idx = (current_idx + 1) % len(self.data)
                    print(f"Retrying with index {current_idx} after error: {str(e)}")
                else:
                    print(f"Failed after {max_retries} attempts. Last error: {str(e)}")
                    
                    # Return a fallback item with blank 3-channel tensor of expected size
                    return torch.zeros(3, 224, 224), "", "", []
    
    @staticmethod
    def custom_collate_fn(batch):
        """
        Custom collate function for batching items with different sized tensors and objects.
        
        Args:
            batch: List of tuples (pixel_values, caption, image_path, weighted_boxes)
        
        Returns:
            Tuple of batched items:
                - pixel_values: Tensor of shape [batch_size, channels, height, width]
                - captions: List of caption strings
                - image_paths: List of image path strings
                - weighted_boxes_batch: List of weighted_boxes dictionaries/tensors
        """
        pixel_values, captions, image_paths, weighted_boxes_batch = zip(*batch)
        
        # Stack the pixel values into a batch tensor
        pixel_values = torch.stack(pixel_values)
        
        # Keep captions, image_paths, and weighted_boxes as lists
        # (they might have different sizes and formats)
        
        return pixel_values, list(captions), list(image_paths), list(weighted_boxes_batch)
    



class CLIPImageDistillation(LightningModule):
    def __init__(self, hparams, clip_model, clip_preprocess):
        super().__init__()
        self.save_hyperparameters(hparams, ignore="clip_model")

        self.student = clip_model
        self.preprocess = clip_preprocess

        self.teacher = PatchTextAggregation(
            similarity_threshold=0.85,
            projection_model_path="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/ImageProjectionModuleDev/TrainedProjectionModule/Cosine&ContrastiveLossImageProjectionModule/PlaceHolder", #proj_module_best.pth
            faiss_index_path="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/trained_models/faiss_clip_index.idx",
            embeddings_json_path="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/trained_models/clip_embeddings.json"
        )

       
         # Load the contrastive-aware teacher weights
        contrastive_teacher_path = "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/models/teacher_contrastive/contrastive_teacher_ViT-L-14_epoch4_val0.0809.pth" #BEST TEACHER: contrastive_teacher_itm_grad_patch_level_epoch4.pth Vit-B-16: C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/models/teacher_contrastive/contrastive_teacher_ViT-16_epoch5_val0.0370.pth
        #C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/models/teacher_contrastive/contrastive_teacher_ViT-L-14_epoch4_val0.0809.pth  C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/models/teacher_contrastive/contrastive_teacher_ViT-16-lowLR_epoch3_val0.3133.pth/PLACEHOLDER
        if os.path.exists(contrastive_teacher_path):
            # Load on same device as model
            state_dict = torch.load(contrastive_teacher_path, map_location=self.device)
            self.teacher.load_state_dict(state_dict, strict=False)
            print(f"Loaded contrastive-aware teacher from {contrastive_teacher_path}")
        else:
            print(f"Warning: Contrastive teacher not found at {contrastive_teacher_path}, using default teacher")
        
        
   
        

        # Transfer ITM head from teacher
        # self.itm_head = copy.deepcopy(self.teacher.itm_head)
        
        # Freeze ITM head parameters for first few epochs (optional)
        # for param in self.itm_head.parameters():
        #    param.requires_grad = False
        # Initialize KNN cache
        self.cache_dir = "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/teacher_cache"
        self.train_knn_cache_path = os.path.join(self.cache_dir, "teacher_100k_train_knn_cache.pkl")
        self.val_knn_cache_path = os.path.join(self.cache_dir, "teacher_10k_val_knn_cache.pkl")  # Update if you have a different name

       
        # Store original CLIP for zero-shot preservation
        #self.original_clip = copy.deepcopy(self.student)
        #for param in self.original_clip.parameters():
        #    param.requires_grad = False
        
        # Load the KNN cache for training data
        if os.path.exists(self.train_knn_cache_path):
            self.teacher.load_caches(knn_cache_path=self.train_knn_cache_path)
            print(f"Loaded training KNN cache")
        else:
            print(f"No KNN cache found at {self.train_knn_cache_path}")
            # Initialize empty KNN cache
            self.teacher.knn_cache = {}



        self.teacher.to(self.device)

        self.teacher.eval()

        # unfreeze the image encoder layers of CLIP
        # Example: Freeze all layers except the final projection layer of the vision encoder.
        for name, param in self.student.vision_model.named_parameters():
            if "proj" not in name:
                param.requires_grad = False

    def forward(self, text=None, image=None):
        if image is not None:
            # Preprocess the image and extract features.
            processed = self.preprocess(images=image, return_tensors="pt")
            pixel_values = processed["pixel_values"].to(self.device)
            image_features = self.student.get_image_features(pixel_values=pixel_values)
            return image_features
        elif text is not None:
            # Preprocess the text and extract features.
            tokens = self.preprocess(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )["input_ids"].to(self.device)
            text_features = self.student.get_text_features(input_ids=tokens)
            return text_features
        else:
            raise ValueError("Either text or image must be provided.")


    """

    # KIND OF HELPS FOR BALANCE
    def compute_contrastive_loss(self, image_embeddings, text_embeddings, temperature=0.05):
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
        batch_size = image_embeddings.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Calculate directional losses separately
        loss_i2t = F.cross_entropy(logits, labels)  
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # Calculate loss ratio to determine imbalance
        ratio = loss_i2t / loss_t2i
        
        # Dynamic weighting based on which direction needs more help
        if ratio > 1.2:  # Image→Text needs more help (20% worse)
            i2t_weight = 1.5
            t2i_weight = 0.5
        elif ratio < 0.83:  # Text→Image needs more help (20% worse)
            i2t_weight = 0.5
            t2i_weight = 1.5
        else:  # Roughly balanced
            i2t_weight = 1.0
            t2i_weight = 1.0
            
        # Apply weights to the losses
        balanced_contrastive_loss = (i2t_weight * loss_i2t + t2i_weight * loss_t2i) / 2.0
        
        # Log the weights for monitoring
        self.log("i2t_weight", i2t_weight)
        self.log("t2i_weight", t2i_weight)
        
        return balanced_contrastive_loss
    
    
    """



    def compute_itm_loss(self, image_embeddings, text_embeddings):
        """Image-Text Matching loss with hard negative mining"""
        batch_size = image_embeddings.shape[0]
        
        # Normalize embeddings for similarity computation
        img_norm = F.normalize(image_embeddings, dim=1)
        txt_norm = F.normalize(text_embeddings, dim=1)
        
        # Compute similarity matrix for hard negative mining
        with torch.no_grad():
            sim_matrix = torch.matmul(img_norm, txt_norm.T)
            sim_matrix.fill_diagonal_(-100)  # Remove positive pairs
            
            # Find hardest negative texts for each image
            _, hard_txt_indices = sim_matrix.topk(1, dim=1)
            
            # Find hardest negative images for each text
            _, hard_img_indices = sim_matrix.T.topk(1, dim=1)
        
        # Process positive pairs (matching image-text)
        pos_features = torch.cat([image_embeddings, text_embeddings], dim=1)
        pos_logits = self.itm_head(pos_features)
        pos_labels = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        # Process hard negative pairs (image with hardest non-matching text)
        neg_txt_embeddings = text_embeddings[hard_txt_indices.squeeze()]
        neg_txt_features = torch.cat([image_embeddings, neg_txt_embeddings], dim=1)
        neg_txt_logits = self.itm_head(neg_txt_features)
        
        # Process hard negative pairs (text with hardest non-matching image)
        neg_img_embeddings = image_embeddings[hard_img_indices.squeeze()]
        neg_img_features = torch.cat([neg_img_embeddings, text_embeddings], dim=1)
        neg_img_logits = self.itm_head(neg_img_features)
        
        # Combine all logits and labels
        itm_logits = torch.cat([pos_logits, neg_txt_logits, neg_img_logits])
        itm_labels = torch.cat([
            pos_labels,
            torch.zeros(batch_size, dtype=torch.long, device=self.device),
            torch.zeros(batch_size, dtype=torch.long, device=self.device)
        ])
        
        return F.cross_entropy(itm_logits, itm_labels)
    #try hard negative mining later
    def compute_contrastive_loss(self, image_embeddings, text_embeddings, temperature=0.05):
        """
        Compute the contrastive loss between image and text embeddings.
        
        Args:
            image_embeddings: Normalized image embeddings [batch_size, embed_dim]
            text_embeddings: Normalized text embeddings [batch_size, embed_dim]
            temperature: Scaling factor for the similarity scores
            
        Returns:
            contrastive_loss: The InfoNCE/NT-Xent contrastive loss
        """
        # Normalize the embeddings along the feature dimension
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        # Calculate the cosine similarity between all possible image-text pairs
        logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
        
        # The positive pairs are along the diagonal
        batch_size = image_embeddings.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Compute the contrastive loss (cross entropy with diagonals as positives)
        loss_i2t = F.cross_entropy(logits, labels)  
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # Average the image-to-text and text-to-image losses
        contrastive_loss = (loss_i2t + loss_t2i) / 2.0
        
        return contrastive_loss
    
    def cosine_distillation_loss(self, student_embeddings, teacher_embeddings):
        """
        Simplified distillation loss without temperature or mixup regularization
        """
        # Normalize embeddings
        student_norm = F.normalize(student_embeddings, dim=1)
        teacher_norm = F.normalize(teacher_embeddings, dim=1)
        
        # Direct cosine similarity (no temperature)
        cos_sim = torch.sum(student_norm * teacher_norm, dim=1)
        
        # Convert to a loss
        return torch.mean(1.0 - cos_sim)
    

    #implement balanced loss
    def training_step(self, batch):
        """
        high level

        take the image and text from the dataset, pass it through the teacher model

        pass the image though the student image encoder

        compute a MSE loss between the student and teacher image embeddings

        return loss
        
        """

        images, captions, image_paths, weighted_boxes_batch = batch

        # ----- IMAGE DISTILLATION -----
        with torch.no_grad():
            teacher_image_embeddings = self.teacher.compute_global_embedding_batch(
                image_paths, captions, weighted_boxes_batch
            ).to(self.device).float()
        student_image_embeddings = self.student.get_image_features(pixel_values=images.to(self.device)).float()
        loss_image = self.cosine_distillation_loss(student_image_embeddings, teacher_image_embeddings)

        # ----- TEXT DISTILLATION ----- 
        teacher_text_embeddings = []
        for caption in captions:
            teacher_text_embeddings.append(self.teacher.text_tokenizer.aggregate_text(caption))
        teacher_text_embeddings = torch.stack(teacher_text_embeddings).to(self.device).float()

        tokens = self.preprocess(
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )["input_ids"].to(self.device)
        student_text_embeddings = self.student.get_text_features(input_ids=tokens).float()

        loss_text = self.cosine_distillation_loss(student_text_embeddings, teacher_text_embeddings)

        # ----- CONTRASTIVE LOSS -----
        contrastive_loss = self.compute_contrastive_loss(
            student_image_embeddings, 
            student_text_embeddings
        )

        # ----- ITM LOSS -----
        #itm_loss = self.compute_itm_loss(student_image_embeddings, student_text_embeddings)
        
        # Start with smaller ITM weight and increase gradually
        #itm_weight = 1.5 #min(1.0, 0.5 + self.current_epoch * 0.1)

        # Add CLIP preservation component
        """
        with torch.no_grad():
            original_clip_embeddings = self.original_clip.get_image_features(
                pixel_values=images.to(self.device)
            ).float()
        
        
        """
        
        #preservation_weight = 0.3  # Start with this, adjust as needed
        #preservation_loss = self.cosine_distillation_loss(
           # student_image_embeddings, original_clip_embeddings
        #)
        
        # ----- COMBINE LOSSES -----
        loss = loss_image + loss_text + 1.0 * contrastive_loss #+ itm_weight * itm_loss + preservation_weight * preservation_loss
        print(f"Image Loss: {loss_image.item():.4f}, Text Loss: {loss_text.item():.4f}, "
            f"Contrastive Loss: {contrastive_loss.item():.4f})")

        self.log("train_loss", loss, prog_bar=True, batch_size=self.hparams.train_batch_size)
        
        return loss

    def validation_step(self, batch):
        images, captions, image_paths, weighted_boxes_batch = batch

        # ----- IMAGE DISTILLATION -----
        with torch.no_grad():
            teacher_image_embeddings = self.teacher.compute_global_embedding_batch(
                image_paths, captions, weighted_boxes_batch
            ).to(self.device).float()
        student_image_embeddings = self.student.get_image_features(pixel_values=images.to(self.device)).float()
        loss_image = self.cosine_distillation_loss(student_image_embeddings, teacher_image_embeddings)

        # ----- TEXT DISTILLATION ----- 
        teacher_text_embeddings = []
        for caption in captions:
            teacher_text_embeddings.append(self.teacher.text_tokenizer.aggregate_text(caption))
        teacher_text_embeddings = torch.stack(teacher_text_embeddings).to(self.device).float()

        tokens = self.preprocess(
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )["input_ids"].to(self.device)
        student_text_embeddings = self.student.get_text_features(input_ids=tokens).float()

        loss_text = self.cosine_distillation_loss(student_text_embeddings, teacher_text_embeddings)

        # ----- CONTRASTIVE LOSS -----
        contrastive_loss = self.compute_contrastive_loss(
            student_image_embeddings, 
            student_text_embeddings
        )

        # ----- ITM LOSS -----
        #itm_loss = self.compute_itm_loss(student_image_embeddings, student_text_embeddings)
        
        # Start with smaller ITM weight and increase gradually
        #itm_weight = 1.5 #min(1.0, 0.5 + self.current_epoch * 0.1)

        # Add CLIP preservation component
        """
        with torch.no_grad():
            original_clip_embeddings = self.original_clip.get_image_features(
                pixel_values=images.to(self.device)
            ).float()
        
        
        """
        
        #preservation_weight = 0.3  # Start with this, adjust as needed
        #preservation_loss = self.cosine_distillation_loss(
           # student_image_embeddings, original_clip_embeddings
        #)
        
        # ----- COMBINE LOSSES -----
        loss = loss_image + loss_text + 1.0 * contrastive_loss #+ itm_weight * itm_loss + preservation_weight * preservation_loss
        print(f"Image Loss: {loss_image.item():.4f}, Text Loss: {loss_text.item():.4f}, "
            f"Contrastive Loss: {contrastive_loss.item():.4f})")

        self.log("val_loss", loss, prog_bar=True, batch_size=self.hparams.eval_batch_size)
        
        return loss

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.total_steps)
        return [optimizer], [scheduler]

    
    def train_dataloader(self) -> DataLoader:
        dataset = MultiModalDataset(self.hparams.train_file, self.preprocess, cache_dir="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/teacher_cache", cache_filename="teacher_100k_train_precache.pkl")
        return DataLoader(dataset, batch_size=self.hparams.eval_batch_size, num_workers=0, pin_memory=True, shuffle=True, persistent_workers=False, collate_fn=MultiModalDataset.custom_collate_fn)
    
    def val_dataloader(self) -> DataLoader:
        if hasattr(self.hparams, 'val_file') and self.hparams.val_file:
            dataset = MultiModalDataset(self.hparams.val_file, self.preprocess, cache_dir="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/teacher_cache", cache_filename="teacher_10k_val_precache.pkl")
            return DataLoader(dataset, batch_size=self.hparams.eval_batch_size, num_workers=0, pin_memory=False, persistent_workers=False, collate_fn=MultiModalDataset.custom_collate_fn)
        return None
    
    def on_validation_epoch_start(self):
        # Switch to validation KNN cache before validation starts
        if os.path.exists(self.val_knn_cache_path):
            self.teacher.load_caches(knn_cache_path=self.val_knn_cache_path)
            print(f"Switched to validation KNN cache")
        else:
            print(f"No validation KNN cache found at {self.val_knn_cache_path}")
        
    def on_validation_epoch_end(self):
        # Switch back to training KNN cache after validation is done
        if os.path.exists(self.train_knn_cache_path):
            self.teacher.load_caches(knn_cache_path=self.train_knn_cache_path)
            print(f"Switched back to training KNN cache")
    
    def on_epoch_start(self):
        """
        if self.current_epoch == 2:
            for param in self.itm_head.parameters():
                param.requires_grad = True
            print("Unfroze ITM head parameters")
        
        """
        
    
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("CLIPImageDistillation")
        parser.add_argument("--train_file", type=str, required=True, help="Path to the training JSON file.")
        parser.add_argument("--val_file", type=str, required=False, default=None, help="Path to the validation JSON file.")
        parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size.")
        parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size.")
        parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
        parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps.")
        parser.add_argument("--total_steps", type=int, default=1000, help="Total training steps.")
        return parent_parser

    def on_epoch_end(self):
        # Handle full resolution switching (existing code)
        if self.current_epoch >= self.hparams.phase1_epochs // 2:
            self.teacher.full_resolution = True
            self.log("phase", "full_resolution") 
        else:
            self.teacher.full_resolution = False
            self.log("phase", "default_resolution")
        
        # Progressive unfreezing based on epoch
        if self.current_epoch == 2:
            # Unfreeze the last encoder block
            for name, param in self.student.vision_model.named_parameters():
                if "encoder.layers.11" in name or "proj" in name:  # Last encoder block (adjust index as needed)
                    param.requires_grad = True
                    print(f"Unfroze layer: {name}")
        
        elif self.current_epoch == 4:
            # Unfreeze more encoder blocks
            for name, param in self.student.vision_model.named_parameters():
                if any(f"encoder.layers.{i}" in name for i in [9, 10, 11]) or "proj" in name:
                    param.requires_grad = True
                    print(f"Unfroze layer: {name}")
        
        elif self.current_epoch == 6:
            # Optionally unfreeze even more layers or the entire model
            for name, param in self.student.vision_model.named_parameters():
                param.requires_grad = True
                
        # Also consider unfreezing the text encoder more progressively if needed
        if self.current_epoch >= 3:
            for param in self.student.text_model.parameters():
                param.requires_grad = True
        
        # Print debug message for full resolution (existing code)
        print(f"Epoch {self.current_epoch}: teacher.full_resolution = {self.teacher.full_resolution}")
        
        # Add debug message to show percentage of unfrozen parameters
        total_params = sum(p.numel() for p in self.student.parameters())
        trainable_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params/total_params:.2%}")
        

        
