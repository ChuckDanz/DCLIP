from image_tokenizer import CLIPPatchTokenizer, TokenizerWithKNN
from text_tokenizer import CLIPTextTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
import pickle
import os


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.text_to_image = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.image_to_text = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm_text = nn.LayerNorm(embed_dim)
        self.norm_image = nn.LayerNorm(embed_dim)
    
    def forward(self, text_embedding, image_embedding):
        """Bidirectional attention between text and image tokens"""
        # text_embedding: [batch_size, num_text_tokens, embed_dim]
        # image_embedding: [batch_size, num_patches, embed_dim]
        
        batch_size = text_embedding.shape[0]
        
        # Text attends to image (sequence to sequence)
        q_text = text_embedding.transpose(0, 1)  # [num_tokens, batch, dim]
        k_img = image_embedding.transpose(0, 1)  # [num_patches, batch, dim]
        v_img = image_embedding.transpose(0, 1)  # [num_patches, batch, dim]
        
        text_output, _ = self.text_to_image(query=q_text, key=k_img, value=v_img)
        text_output = text_output.transpose(0, 1)  # [batch, num_tokens, dim]
        text_output = self.norm_text(text_embedding + text_output)
        
        # Image attends to text (sequence to sequence)
        q_img = image_embedding.transpose(0, 1)  # [num_patches, batch, dim]
        k_text = text_embedding.transpose(0, 1)  # [num_tokens, batch, dim]
        v_text = text_embedding.transpose(0, 1)  # [num_tokens, batch, dim]
        
        image_output, _ = self.image_to_text(query=q_img, key=k_text, value=v_text)
        image_output = image_output.transpose(0, 1)  # [batch, num_patches, dim]
        image_output = self.norm_image(image_embedding + image_output)
        
        return text_output, image_output


class PatchTextAggregation(nn.Module):
    def __init__(self, 
                 embed_dim=512,  #FIX THIS IF GOING FROM VIT L TO VIT B
                 num_heads=8, 
                 similarity_threshold=0.85,
                 projection_model_path=None,
                 faiss_index_path=None,
                 embeddings_json_path=None):
        super(PatchTextAggregation, self).__init__()
        self.embed_dim = embed_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_tokenizer = CLIPTextTokenizer()
        self.patch_tokenizer = CLIPPatchTokenizer()
        self.cross_modal_attention = CrossModalAttention(embed_dim, num_heads).to(self.device)

        self.knn_cache = {}

        """
         self.itm_head = nn.Sequential(
            nn.Linear(512*2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 2)
        ).to(self.device)
        
        
        """
       
        # Initialize advanced image tokenization capabilities (KNN + Projection)
        self.use_knn_projection = all([
            projection_model_path,
            faiss_index_path, 
            embeddings_json_path
        ])
        
        if self.use_knn_projection:
            print("Initializing KNN+Projection capabilities for image tokenization")
            self.advanced_tokenizer = TokenizerWithKNN(
                projection_model_path=projection_model_path,
                faiss_index_path=faiss_index_path,
                embeddings_json_path=embeddings_json_path,
                similarity_threshold=similarity_threshold,
                device=self.device
            )
            self.similarity_threshold = similarity_threshold
        else:
            print("KNN+Projection disabled: missing required paths")
            self.advanced_tokenizer = None
        

        self.full_resolution = False


    #function returns a tuple of (text_embedding, patch_embedding, cosine similarity) for each text embedding

    def load_caches(self, knn_cache_path=None):
        """
        Load KNN cache from disk
        
        Args:
            knn_cache_path: Path to the KNN cache pickle file
        """
        if knn_cache_path and os.path.exists(knn_cache_path):
            try:
                with open(knn_cache_path, 'rb') as f:
                    self.knn_cache = pickle.load(f)
                print(f"Loaded KNN cache with {len(self.knn_cache)} entries")
            except Exception as e:
                print(f"Error loading KNN cache: {e}")
                self.knn_cache = {}
        else:
            print(f"No KNN cache found at {knn_cache_path}")
            self.knn_cache = {}
        
        return self


    def compute_patch_text_similarity(self, image_path, text, bb_properties=None):
        """
        Vectorized computation of similarity between text and image patches.
        Returns a list of tuples (text_embedding, (patch_embedding, weight), cosine_similarity)
        """
        # Get text embeddings as a tensor: [n_text, embed_dim]
        chunked_text_embeddings = self.text_tokenizer.get_embeddings(text)
        if not chunked_text_embeddings:
            return []
        text_embeddings = torch.stack(chunked_text_embeddings)  # shape: [n_text, d]
        
        # Use precomputed bounding boxes if available
        if bb_properties is None:
            bb_properties = self.patch_tokenizer.get_weighted_bounding_boxes(image_path)
        
        image = Image.open(image_path)
        patch_list = self.patch_tokenizer.encode_weighted_bounding_boxes(image, bb_properties)  # list of (embedding, weight)
        if not patch_list:
            return []
        
        # Stack patch embeddings: [n_patch, embed_dim]
        patch_embeddings = torch.stack([pe[0] for pe in patch_list])
        
        # Normalize embeddings to compute cosine similarity via dot products
        norm_text = torch.nn.functional.normalize(text_embeddings, dim=1)
        norm_patch = torch.nn.functional.normalize(patch_embeddings, dim=1)
        
        # Compute similarity matrix: shape [n_text, n_patch]
        sim_matrix = torch.matmul(norm_text, norm_patch.transpose(0, 1))
        
        # For each patch, find the text with the maximum similarity
        max_sim, best_indices = sim_matrix.max(dim=0)  # both of shape [n_patch]
        
        caption_text_pairs = []
        for i, (patch_emb, weight) in enumerate(patch_list):
            best_text_emb = text_embeddings[best_indices[i]]
            sim = max_sim[i].item()
            caption_text_pairs.append((best_text_emb, (patch_emb, weight), sim))
        
        return caption_text_pairs


    def compute_image_patch_weight(self, image_path, text, bb_properties=None):
        """
        Computes the weight for each patch based on the area of the bounding box,
        Args:
            image_path (str): Path to the input image.
            text (str): Input text for similarity computation.
        Returns:
            list: A list of tuples (text_embedding, patch_embedding, cosine similarity).
        """
        
        if bb_properties is None:
            bb_properties = self.patch_tokenizer.get_weighted_bounding_boxes(image_path)

        similarity = self.compute_patch_text_similarity(image_path, text, bb_properties)
        areas = []
        confidences = []
        similarities = []

        for box, conf in bb_properties:
            x1, y1, x2, y2 = box
            areas.append((x2 - x1) * (y2 - y1))
            confidences.append(conf)

        for pair in similarity:
            similarities.append(pair[2])
        
        weights = [a * c * s for a, c, s in zip(areas, confidences, similarities)]
        if not weights:  # ensure weights list is not empty
            return []

        total = sum(weights)
        if total == 0:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        else:
            normalized_weights = [w / total for w in weights]

        return normalized_weights
        
    def weighted_text_image(self, embeddings, weights):

        """
        This function computes the weighted image pairs
        args:
        embeddings - tuple(text_embedding, patch_embedding, cosine similarity)
        weights - list(weights)
        Returns: a list of tuples (text_embedding, patch_embedding * weight)
        """

        pairs = [] #tuple(text, patch embedding)
        #the zip function in python is how you loop through two lists at one time

        for embedding, weight in zip(embeddings, weights):
            text_embedding, patch_embedding, _ = embedding
            pairs.append((text_embedding, patch_embedding[0] * weight))

        return pairs
    

    #this is to reinforce patch-text relationship
    def cross_attention(self, text_embedding, patch_embedding):
        """
        Now handles bidirectional attention between text and image
        
        Args: 
            text_embedding - text embeddings 
            patch_embedding - patch embeddings
        
        Returns: attended text embeddings (for backward compatibility)
        """
        # Get both attended outputs
        attended_text, attended_image = self.cross_modal_attention(text_embedding, patch_embedding)
        
        # Return both embeddings directly as a tuple
        return attended_text, attended_image
    
    def aggregation(self, attended_text, temperature=2.0):
        """
        Temperature-scaled aggregation that amplifies important patches
        Lower temperature = sharper focus on important patches
        """
        batch_size, seq_len, embed_dim = attended_text.shape
        
        # Get importance via similarity to mean embedding (reference point)
        mean_embed = torch.mean(attended_text, dim=1, keepdim=True)
        similarities = F.cosine_similarity(
            attended_text,
            mean_embed.expand(-1, seq_len, -1),
            dim=2
        )
        
        # Apply temperature scaling - lower temp = sharper focus
        weights = F.softmax(similarities / temperature, dim=1)
        
        # Weight and sum
        weights = weights.unsqueeze(-1).expand(-1, -1, embed_dim)
        global_embedding = torch.sum(attended_text * weights, dim=1)
        
        return global_embedding
    
    #STACK TENSOR INTO ONE
    def compute_global_embedding_batch(self, image_paths, texts, weighted_boxes_batch=None):
        """
        Compute global embeddings for a batch of images and texts.
        
        Args:
            image_paths (list): List of paths to images.
            texts (list): List of text captions.
            weighted_boxes_batch (dict or list, optional): Pre-computed bounding boxes.
            
        Returns:
            torch.Tensor: Global embedding tensor.
        """
        device = self.device
        
        # Handle weighted_boxes_batch in different formats
        if hasattr(self, 'yolo_cache') and weighted_boxes_batch is None:
            print("Using cached YOLO detections")
            weighted_boxes_batch = {path: self.yolo_cache.get(path, []) for path in image_paths}
        elif weighted_boxes_batch is not None and isinstance(weighted_boxes_batch, list):
            weighted_boxes_batch = dict(zip(image_paths, weighted_boxes_batch))
        elif weighted_boxes_batch is None:
            weighted_boxes_batch = self.patch_tokenizer.get_weighted_bounding_boxes_batch(image_paths)
            if isinstance(weighted_boxes_batch, list):
                weighted_boxes_batch = dict(zip(image_paths, weighted_boxes_batch))
        
        patch_embeddings_list = []
        text_embeddings_list = []
        max_patches = 0

        for image_path, text in zip(image_paths, texts):
            # Safe image loading
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                image = Image.new("RGB", (224, 224))
            
            wboxes = weighted_boxes_batch.get(image_path, [])
            
            # Process patches using KNN/projection or standard CLIP
            if self.use_knn_projection and self.advanced_tokenizer is not None and wboxes:
                patch_embed_list = []
                
                for (box, conf) in wboxes:
                    try:
                        x1, y1, x2, y2 = box
                        w, h = image.size
                        position = [x1/w, y1/h, x2/w, y2/h]
                        patch = image.crop((x1, y1, x2, y2))
                        
                        # Create cache key
                        try:
                            import hashlib
                            patch_bytes = np.array(patch).tobytes()[:1000]
                            position_str = f"{x1/w:.4f}_{y1/h:.4f}_{x2/w:.4f}_{y2/h:.4f}"
                            cache_key = hashlib.md5((patch_bytes + position_str.encode())).hexdigest()
                        except Exception as e:
                            print(f"Error creating cache key: {e}")
                            cache_key = None
                        
                        # Try cache lookup
                        embedding = None
                        source = None
                        similarity = 0.0
                        
                        if cache_key and hasattr(self, 'knn_cache') and cache_key in self.knn_cache:
                            try:
                                cached_result = self.knn_cache[cache_key]
                                embedding = cached_result['embedding']
                                source = cached_result['source']
                                similarity = cached_result['similarity']
                                #print(f"Using cached {source} embedding: similarity={similarity:.3f}")
                            except Exception as e:
                                print(f"Error using cache: {e}")
                                embedding = None
                        
                        # Compute embedding if not in cache
                        if embedding is None:
                            try:
                                embedding, source, similarity = self.advanced_tokenizer.knn_or_projection(patch, position)
                                print(f"Using {source} embedding: similarity={similarity:.3f}")
                                
                                # Save to cache
                                if cache_key and hasattr(self, 'knn_cache'):
                                    try:
                                        if hasattr(embedding, 'tolist'):
                                            serialized_embedding = embedding.tolist()
                                        else:
                                            serialized_embedding = embedding
                                        
                                        self.knn_cache[cache_key] = {
                                            'embedding': serialized_embedding,
                                            'source': source,
                                            'similarity': similarity,
                                            'image_path': image_path
                                        }
                                    except Exception as e:
                                        print(f"Error saving to cache: {e}")
                            except Exception as e:
                                print(f"Error in knn_or_projection: {e}")
                                try:
                                    # Fall back to direct CLIP
                                    clip_tokenizer = self.patch_tokenizer
                                    processed = clip_tokenizer.clip_preprocess(images=patch, text="", return_tensors="pt")
                                    pixel_values = processed['pixel_values'].to(device)
                                    
                                    with torch.no_grad():
                                        image_features = clip_tokenizer.clip_model.get_image_features(pixel_values=pixel_values)
                                    
                                    embedding = image_features.cpu().numpy().flatten()
                                    source = "direct_clip"
                                    similarity = 0.0
                                    print(f"Using direct CLIP embedding")
                                except Exception as e2:
                                    print(f"Error in direct CLIP embedding: {e2}")
                                    embedding = np.zeros(512, dtype=np.float32)
                                    source = "error"
                                    similarity = 0.0
                        
                        # Process embedding to ensure it's in correct format
                        try:
                            # Handle dictionary
                            if isinstance(embedding, dict):
                                print(f"Converting dictionary with keys {list(embedding.keys())}")
                                
                                # Try standard keys
                                for key in ['embedding', 'features', 'vector', 'representation', 'values', 'data']:
                                    if key in embedding:
                                        if isinstance(embedding[key], dict):
                                            # Handle nested dictionary
                                            for subkey in ['embedding', 'features', 'vector', 'values']:
                                                if subkey in embedding[key]:
                                                    try:
                                                        embedding = np.array(embedding[key][subkey], dtype=np.float32).flatten()
                                                        break
                                                    except:
                                                        continue
                                        else:
                                            try:
                                                embedding = np.array(embedding[key], dtype=np.float32).flatten()
                                                break
                                            except:
                                                continue
                                
                                # If still a dict, use values
                                if isinstance(embedding, dict):
                                    try:
                                        values = list(embedding.values())
                                        if all(isinstance(v, (int, float)) for v in values):
                                            embedding = np.array(values, dtype=np.float32).flatten()
                                        else:
                                            # Try first value that looks like an embedding
                                            for v in values:
                                                if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                                                    embedding = np.array(v, dtype=np.float32).flatten()
                                                    break
                                            else:
                                                # No suitable value found
                                                embedding = np.zeros(512, dtype=np.float32)
                                    except:
                                        embedding = np.zeros(512, dtype=np.float32)
                            
                            # Convert to numpy array if not already
                            if not isinstance(embedding, np.ndarray):
                                embedding = np.array(embedding, dtype=np.float32).flatten()
                            
                            # Handle numpy object arrays
                            if embedding.dtype == np.dtype('O'):
                                try:
                                    if embedding.size > 0 and hasattr(embedding[0], '__iter__'):
                                        # Extract first item if it's a sequence
                                        embedding = np.array(embedding[0], dtype=np.float32).flatten()
                                    else:
                                        # Convert to list then to float array
                                        embedding = np.array(embedding.tolist(), dtype=np.float32).flatten()
                                except:
                                    embedding = np.zeros(512, dtype=np.float32)
                            
                            # Ensure we have a proper float32 array
                            embedding = np.array(embedding, dtype=np.float32).flatten()
                            
                            # Final safety check - no NaNs or empty arrays
                            if embedding.size == 0:
                                embedding = np.zeros(512, dtype=np.float32)
                            else:
                                try:
                                    if np.isnan(embedding).any():
                                        embedding = np.zeros(512, dtype=np.float32)
                                except:
                                    # If isnan fails, check another way
                                    if not np.all(np.isfinite(embedding)):
                                        embedding = np.zeros(512, dtype=np.float32)
                        except Exception as e:
                            print(f"Error processing embedding: {e}")
                            embedding = np.zeros(512, dtype=np.float32)
                        
                        # Convert to tensor
                        try:
                            embedding_tensor = torch.tensor(embedding, device=device).unsqueeze(0)
                        except Exception as e:
                            print(f"Error converting to tensor: {e}")
                            embedding_tensor = torch.zeros((1, 512), device=device)
                        
                        # Add to embeddings list
                        patch_embed_list.append((embedding_tensor, conf))
                    
                    except Exception as box_e:
                        print(f"Error processing box {box}: {box_e}")
                        # Skip this box and continue
                
            else:
                # Use original patch tokenization
                try:
                    patch_embed_list = self.patch_tokenizer.encode_weighted_bounding_boxes(
                        image, wboxes, full_resolution=self.full_resolution
                    )
                except Exception as e:
                    print(f"Error in encode_weighted_bounding_boxes: {e}")
                    patch_embed_list = []
            
            # Handle empty patch list
            if not patch_embed_list:
                print(f"Warning: No patch embeddings for {image_path}. Using zeros.")
                patch_tensor = torch.zeros((1, self.embed_dim), device=device)
            else:
                try:
                    # Stack embeddings
                    patch_tensors = []
                    for pe_tuple in patch_embed_list:
                        try:
                            pe = pe_tuple[0]
                            if not torch.isfinite(pe).all():
                                pe = torch.zeros_like(pe)
                            
                            # Ensure correct shape: should be 1D or 2D
                            if pe.dim() > 2:
                                pe = pe.reshape(-1, pe.shape[-1])
                            elif pe.dim() == 1:
                                pe = pe.unsqueeze(0)
                                
                            patch_tensors.append(pe)
                        except Exception as e:
                            print(f"Error processing patch tensor: {e}")
                            continue
                    
                    if not patch_tensors:
                        patch_tensor = torch.zeros((1, self.embed_dim), device=device)
                    else:
                        # Standardize shapes before stacking
                        for i, pt in enumerate(patch_tensors):
                            if pt.shape[-1] != self.embed_dim:
                                print(f"Fixing tensor dimension mismatch: {pt.shape}")
                                patch_tensors[i] = torch.zeros((1, self.embed_dim), device=device)
                        
                        patch_tensor = torch.cat(patch_tensors, dim=0).to(device)
                except Exception as e:
                    print(f"Error stacking patch embeddings: {e}")
                    patch_tensor = torch.zeros((1, self.embed_dim), device=device)
            
            # Add to embedding lists
            num_patches = patch_tensor.shape[0]
            max_patches = max(max_patches, num_patches)
            patch_embeddings_list.append(patch_tensor)
            
            # Process text
            try:
                text_emb_tokens = self.text_tokenizer.get_embeddings(text, return_token_level=True)
                if isinstance(text_emb_tokens, list):
                    if len(text_emb_tokens) > 0:
                        text_emb = torch.stack(text_emb_tokens)
                    else:
                        text_emb = torch.zeros((1, self.embed_dim), device=self.device)
                else:
                    text_emb = text_emb_tokens
                if torch.isnan(text_emb).any() or torch.isinf(text_emb).any():
                    print(f"Warning: NaN/Inf in text embedding for: {text}")
                    text_emb = torch.zeros_like(text_emb)
                
                # Standardize to [1, embed_dim]
                if text_emb.dim() == 1:
                    text_emb = text_emb.unsqueeze(0)
                
                text_embeddings_list.append(text_emb.to(device))
            except Exception as e:
                print(f"Error processing text embedding: {e}")
                text_embeddings_list.append(torch.zeros((1, self.embed_dim), device=device))
        
        # Pad all patch embeddings to the same length - FIXED VERSION
        padded_patches = []
        for i, p in enumerate(patch_embeddings_list):
            try:
                # Ensure p is 2D: [num_patches, embed_dim]
                if p.dim() == 3:
                    print(f"Tensor at index {i} has 3 dims with shape {p.shape}, squeezing...")
                    p = p.squeeze(1)
                elif p.dim() == 1:
                    print(f"Tensor at index {i} has 1 dim with shape {p.shape}, expanding...")
                    p = p.unsqueeze(0)
                    
                # Get current dimensions
                num_patches = p.shape[0]
                embed_dim = p.shape[1]
                
                # Pad to max_patches
                if num_patches < max_patches:
                    pad = torch.zeros((max_patches - num_patches, embed_dim), device=device)
                    p = torch.cat([p, pad], dim=0)
                elif num_patches > max_patches:
                    p = p[:max_patches]
                    
                padded_patches.append(p)
            except Exception as e:
                print(f"Error padding patches for index {i}: {e}")
                padded_patches.append(torch.zeros((max_patches, self.embed_dim), device=device))
        
        # Stack final tensors - FIXED VERSION
        try:
            # Ensure all padded_patches have same shape before stacking
            for i, p in enumerate(padded_patches):
                if p.dim() != 2 or p.shape != (max_patches, self.embed_dim):
                    print(f"Fixing tensor at index {i}: shape {p.shape}")
                    padded_patches[i] = torch.zeros((max_patches, self.embed_dim), device=device)
            
            # Ensure text embeddings are uniform shape
            for i, t in enumerate(text_embeddings_list):
                if t.dim() == 1:
                    text_embeddings_list[i] = t.unsqueeze(0)
                elif t.dim() > 2:
                    print(f"Text embedding at {i} has unexpected shape: {t.shape}")
                    text_embeddings_list[i] = t.view(1, -1)[:, :self.embed_dim]
                
                # Ensure correct dimension
                if text_embeddings_list[i].shape[1] != self.embed_dim:
                    print(f"Text embedding dimension mismatch: {text_embeddings_list[i].shape}")
                    text_embeddings_list[i] = torch.zeros((1, self.embed_dim), device=device)
                
            batch_patches = torch.stack(padded_patches)

            max_tokens = max(t.shape[0] for t in text_embeddings_list)
            padded_text = []

            for text_emb in text_embeddings_list:
                num_tokens = text_emb.shape[0]
                if num_tokens < max_tokens:
                    # Pad with zeros
                    padding = torch.zeros((max_tokens - num_tokens, self.embed_dim), device=device)
                    text_emb = torch.cat([text_emb, padding], dim=0)
                elif num_tokens > max_tokens:
                    # Truncate
                    text_emb = text_emb[:max_tokens]
                padded_text.append(text_emb)

            batch_text = torch.stack(padded_text)
            


            
            print(f"Final batch shapes: text {batch_text.shape}, patches {batch_patches.shape}")
        except Exception as e:
            print(f"Error stacking final tensors: {e}")
            batch_size = len(image_paths)
            batch_patches = torch.zeros((batch_size, max_patches, self.embed_dim), device=device)
            batch_text = torch.zeros((batch_size, 1, self.embed_dim), device=device)
        
        try:
            # Update this line to capture both outputs
            attended_text, attended_image = self.cross_attention(batch_text, batch_patches)
        except Exception as e:
            print(f"Error in cross attention: {e}")
            attended_text = batch_text
            attended_image = batch_patches

        # Global aggregation for both text and image
        try:
            # Process both attended representations
            text_global = self.aggregation(attended_text)
            image_global = self.aggregation(attended_image)
            
            # Combine for balanced representation (this is key for improving both directions)
            global_embedding = 0.5 * text_global + 0.5 * image_global
            
            if torch.isnan(global_embedding).any() or torch.isinf(global_embedding).any():
                print("Warning: NaN/Inf in global embedding")
                global_embedding = torch.zeros_like(global_embedding)
        except Exception as e:
            print(f"Error in aggregation: {e}")
            global_embedding = torch.zeros((len(image_paths), self.embed_dim), device=device)

        return global_embedding

        #FIXED RETRAIN