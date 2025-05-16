import os
import torch
import pickle
import argparse
import dbm
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_lightning import seed_everything

# Import your models and dataset
from patch_text_aggregation import PatchTextAggregation
from CLIP_image_distillation import MultiModalDataset

import torch.nn as nn

# Memory-efficient DBM-based cache
class DBMCache:
    def __init__(self, path):
        """Initialize a dictionary-like object that uses DBM file for storage"""
        self.path = path
        self.db = dbm.open(path, 'c')  # Create or open existing
        self._keys = None  # Lazy-loaded keys
        self.dirty_keys = set()  # Track keys that need to be saved
        
    def __getitem__(self, key):
        """Get an item from the cache"""
        try:
            if isinstance(key, bytes):
                k = key
            else:
                k = key.encode('utf-8')
                
            if k in self.db:
                return json.loads(self.db[k].decode('utf-8'))
            return None
        except Exception as e:
            print(f"Cache read error for {key}: {e}")
            return None
        
    def __setitem__(self, key, value):
        """Add/update an item in the cache"""
        try:
            if isinstance(key, bytes):
                k = key
            else:
                k = key.encode('utf-8')
                
            self.db[k] = json.dumps(value).encode('utf-8')
            self.dirty_keys.add(key)
            if self._keys is not None:
                self._keys.add(key)
        except Exception as e:
            print(f"Cache write error for {key}: {e}")
        
    def __contains__(self, key):
        """Check if key exists in cache"""
        try:
            if isinstance(key, bytes):
                return key in self.db
            return key.encode('utf-8') in self.db
        except:
            return False
        
    def __len__(self):
        """Count entries in cache"""
        if self._keys is None:
            try:
                # This can be expensive, so we cache the result
                self._keys = set(k.decode('utf-8') for k in self.db.keys())
            except:
                # If there's an error, return a minimal estimation
                return sum(1 for _ in self.db.keys())
        return len(self._keys)
    
    def keys(self):
        """Get all keys"""
        if self._keys is None:
            self._keys = set(k.decode('utf-8') for k in self.db.keys())
        return self._keys
    
    def items(self):
        """Iterate through (key, value) pairs"""
        for k in self.keys():
            yield k, self[k]
    
    def sync(self):
        """Ensure all changes are written to disk"""
        self.db.sync()
    
    def close(self):
        """Close the database"""
        self.db.sync()
        self.db.close()


def main(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize your teacher model
    teacher = PatchTextAggregation(
        similarity_threshold=0.85,
        projection_model_path="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/ImageProjectionModuleDev/TrainedProjectionModule/PlaceHolder", #Cosine&ContrastiveLossImageProjectionModule/proj_module_best.pth
        faiss_index_path="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/trained_models/PlaceHolder", #faiss_clip_index.idx
        embeddings_json_path="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/TokenizerCLIP/trained_models/clip_embeddings.json"
    ).to(device)

    print("\n=== CLIP MODEL DEBUG INFO ===")
    if hasattr(teacher, 'clip_model'):
        config = teacher.clip_model.config
        print(f"CLIP model type: {config.model_type}")
        print(f"Vision model: {config.vision_config.model_type}")
        print(f"Patch size: {config.vision_config.patch_size}")
        print(f"Embedding dimension: {config.projection_dim}")
        print("=" * 30 + "\n")
    
    
    # Print model structure for debugging
    print("Teacher model components:")
    for name, module in teacher.named_children():
        print(f"- {name}: {type(module).__name__}")
    
    # Freeze all parameters by default
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Unfreeze just the cross-attention and projection components
    for name, param in teacher.named_parameters():
        # Look for cross-attention, projection, or final layer parameters
        if any(key in name for key in ['cross_attn', 'attention', 'proj', 'fusion', 'final']):
            param.requires_grad = True
            print(f"Unfreezing: {name}")


    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print(f"Training {trainable_params:,} parameters out of {sum(p.numel() for p in teacher.parameters()):,}")
    
    # Set up KNN caching with memory-mapped DBM
    cache_dir = "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/teacher_cache"
    train_knn_cache_path = os.path.join(cache_dir, "teacher_100k_train_knn_cache.pkl")
    dbm_path = os.path.join(cache_dir, "teacher_100k_train_knn_cache.dbm")
    os.makedirs(cache_dir, exist_ok=True)

    # Load or create the memory-mapped KNN cache
    if hasattr(teacher, 'knn_cache'):
        # Check if we need to convert from pickle to DBM format
        if os.path.exists(train_knn_cache_path) and not os.path.exists(dbm_path + ".db"):
            try:
                print(f"Converting pickle cache to DBM format (one-time operation)...")
                # Get file size to check if we need chunking
                pickle_size = os.path.getsize(train_knn_cache_path)
                print(f"Pickle cache size: {pickle_size/1024/1024:.1f} MB")
                
                # Create DBM database
                db = dbm.open(dbm_path, 'n')  # Create new file
                
                # Load and convert in one go if small enough, otherwise chunk it
                if pickle_size < 500 * 1024 * 1024:  # Less than 500MB
                    with open(train_knn_cache_path, 'rb') as f:
                        pickle_cache = pickle.load(f)
                        
                    print(f"Converting {len(pickle_cache)} entries to DBM format...")
                    count = 0
                    for key, value in pickle_cache.items():
                        db[key.encode('utf-8')] = json.dumps(value).encode('utf-8')
                        count += 1
                        if count % 10000 == 0:
                            print(f"Converted {count} entries...")
                            db.sync()  # Periodically sync to avoid memory buildup
                    
                    db.close()
                    print(f"Successfully converted to DBM format at {dbm_path}")
                else:
                    print(f"Cache too large for direct conversion, will build incrementally during training")
                    db.close()
            except Exception as e:
                print(f"Error converting cache: {e}")
                # We'll start with empty cache if conversion fails
        
        # Now open the DBM cache (whether newly converted or existing)
        try:
            teacher.knn_cache = DBMCache(dbm_path)
            print(f"Using memory-efficient KNN cache with {len(teacher.knn_cache)} entries")
        except Exception as e:
            print(f"Error opening DBM cache: {e}, using empty cache")
            teacher.knn_cache = {}
    else:
        teacher.knn_cache = {}
        print("Model doesn't support KNN cache")
    
    # Create dataset and dataloader with caching
    from transformers import CLIPProcessor
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    # Training dataset
    dataset = MultiModalDataset(
        args.train_file, 
        clip_processor, 
        cache_dir="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/teacher_cache", 
        cache_filename="teacher_100k_train_precache.pkl"
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        collate_fn=MultiModalDataset.custom_collate_fn
    )
    
    # Validation dataset
    val_file = args.train_file.replace("_train.json", "_val.json")
    if not os.path.exists(val_file):
        print(f"Warning: Validation file {val_file} not found. Using default 10k validation file.")
        val_file = "C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/teacher_dataset/teacher_10k_val.json"
        
    print(f"Using validation file: {val_file}")
    
    val_dataset = MultiModalDataset(
        val_file,
        clip_processor,
        cache_dir="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/teacher_cache",
        cache_filename="teacher_10k_val_precache.pkl"
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=MultiModalDataset.custom_collate_fn
    )
    print(f"Training set size: {len(dataset)} samples")
    print(f"Validation set size: {len(val_dataset)} samples")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        [p for p in teacher.parameters() if p.requires_grad], 
        lr=args.learning_rate
    )
    
    # Contrastive loss function
    def compute_contrastive_loss(image_embeddings, text_embeddings, temperature=0.05):
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
        batch_size = image_embeddings.shape[0]
        labels = torch.arange(batch_size, device=device)
        
        loss_i2t = F.cross_entropy(logits, labels)  
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2.0
    
    
    # Validation function
    def validate():
        teacher.eval()
        val_contrastive_loss = 0.0
        val_combined_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                images, captions, image_paths, weighted_boxes_batch = batch
                
                # Get embeddings from teacher
                img_emb = teacher.compute_global_embedding_batch(
                    image_paths, captions, weighted_boxes_batch
                ).to(device)
                
                txt_emb = []
                for caption in captions:
                    txt_emb.append(teacher.text_tokenizer.aggregate_text(caption))
                txt_emb = torch.stack(txt_emb).to(device)
                
                # Compute losses
                c_loss = compute_contrastive_loss(img_emb, txt_emb)
                combined = c_loss 
                
                # Accumulate losses
                val_contrastive_loss += c_loss.item()
                val_combined_loss += combined.item()
                num_batches += 1
        
        # Calculate average losses
        val_contrastive_loss /= num_batches
        val_combined_loss /= num_batches
        
        return {
            "contrastive": val_contrastive_loss,
            "combined": val_combined_loss
        }
    
    # Function to save KNN cache - modified for DBM cache
    def save_knn_cache():
        if hasattr(teacher, 'knn_cache'):
            # Check if we're using DBM cache
            if hasattr(teacher.knn_cache, 'sync'):
                print("Syncing DBM cache to disk...")
                teacher.knn_cache.sync()
                print("DBM cache synced successfully")
            else:
                # Old-style pickle cache saving
                print(f"Saving regular KNN cache with {len(teacher.knn_cache)} entries")
                temp_path = train_knn_cache_path + ".tmp"
                try:
                    with open(temp_path, 'wb') as f:
                        pickle.dump(teacher.knn_cache, f)
                    
                    if os.path.exists(temp_path):
                        if os.path.exists(train_knn_cache_path):
                            os.remove(train_knn_cache_path)
                        os.rename(temp_path, train_knn_cache_path)
                        print(f"Successfully saved KNN cache to {train_knn_cache_path}")
                except Exception as e:
                    print(f"Error saving KNN cache: {e}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Training loop with error handling
    best_val_loss = float('inf')
    try:
        for epoch in range(args.epochs):
            # ===== TRAINING PHASE =====
            teacher.train()
            epoch_loss = 0.0
            
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training")):
                images, captions, image_paths, weighted_boxes_batch = batch
                
                # Zero gradients at the start of each batch
                optimizer.zero_grad()
                
                # Get embeddings from teacher
                img_emb = teacher.compute_global_embedding_batch(
                    image_paths, captions, weighted_boxes_batch
                ).to(device)
                
                txt_emb = []
                for caption in captions:
                    txt_emb.append(teacher.text_tokenizer.aggregate_text(caption))
                txt_emb = torch.stack(txt_emb).to(device)
                
                # Compute contrastive loss
                loss = compute_contrastive_loss(img_emb, txt_emb)
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Log progress (every 10 batches to avoid console spam)
              
                print(f"Batch {batch_idx}, Contrastive Loss: {loss.item():.4f}")
                
                # Track loss
                
                # Save KNN cache periodically
                if hasattr(teacher, 'knn_cache') and batch_idx > 0 and batch_idx % 100 == 0:
                    save_knn_cache()
            
            # Save KNN cache at end of epoch
            if hasattr(teacher, 'knn_cache'):
                save_knn_cache()
            
            # Calculate average training loss
            avg_loss = epoch_loss / len(dataloader)
            
            print(f"Epoch {epoch+1} Training Stats:")
            print(f"  Average Loss: {avg_loss:.4f}")
            
            # ===== VALIDATION PHASE =====
            print("\nRunning validation...")
            val_losses = validate()
            
            print(f"Epoch {epoch+1} Validation Stats:")
            print(f"  Validation Loss: {val_losses['combined']:.4f}")
            
            # Save the model with validation loss in filename
            epoch_save_path = f"{args.output_path.rsplit('.', 1)[0]}_epoch{epoch+1}_val{val_losses['combined']:.4f}.pth"
            torch.save(teacher.state_dict(), epoch_save_path)
            print(f"Saved epoch model to {epoch_save_path}")
            
            # Save the best model based on validation loss
            if val_losses['combined'] < best_val_loss:
                best_val_loss = val_losses['combined']
                torch.save(teacher.state_dict(), args.output_path)
                print(f"🏆 New best model saved with validation loss {best_val_loss:.4f}")
    
    except KeyboardInterrupt:
        print("Training interrupted. Saving final model and cache...")
        if hasattr(teacher, 'knn_cache'):
            save_knn_cache()
            if hasattr(teacher.knn_cache, 'close'):
                teacher.knn_cache.close()
        torch.save(teacher.state_dict(), args.output_path + ".interrupt.pth")
        print("Saved interrupted model state.")
    
    except Exception as e:
        print(f"Error during training: {e}")
        if hasattr(teacher, 'knn_cache'):
            save_knn_cache()
            if hasattr(teacher.knn_cache, 'close'):
                teacher.knn_cache.close()
        torch.save(teacher.state_dict(), args.output_path + ".error.pth")
        raise

    # Clean up at the end
    if hasattr(teacher, 'knn_cache') and hasattr(teacher.knn_cache, 'close'):
        teacher.knn_cache.close()
    
    print("Training completed successfully!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Contrastive-Aware Teacher with Gradient Accumulation")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSON file")
    parser.add_argument("--val_file", type=str, default="C:/Users/Daniel Csizmadia/Desktop/TokenizerCLIP/teacher_dataset/teacher_10k_val.json", help="Path to validation JSON file (optional)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per accumulation step")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--output_path", type=str, default="./teacher_contrastive/contrastive_teacher_ViT-16.pth", 
                        help="Path to save the trained teacher model")
    args = parser.parse_args()
    main(args)