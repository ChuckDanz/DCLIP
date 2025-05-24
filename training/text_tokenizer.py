import string
import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPModel, BertModel, BertTokenizer
from text_projection_module import ProjectionModule
import os
import gensim.downloader as api

class CLIPTextTokenizer:
    def __init__(self, 
                 tokenizer_path="openai/clip-vit-base-patch16", #openai/clip-vit-base-patch16
                 bert_model_name="bert-base-uncased",
                 max_chunk_size=77, 
                 device="cpu",
                 projection_path= "/PlaecHolder", #projection_model_weighted.pth
                 complexity_threshold=0.35):
        """
        Initializes the CLIP tokenizer with complexity scoring.
        """
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.model = CLIPModel.from_pretrained(tokenizer_path).to(device)
        self.max_chunk_size = max_chunk_size
        self.device = device
        self.complexity_threshold = complexity_threshold

        # For BERT embedding
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(device)

        self.projection = ProjectionModule(bert_dim=768, clip_dim=512).to(device)

        # Load projection model
        try:
            state_dict = torch.load(projection_path, map_location=device)
            self.projection.load_state_dict(state_dict)
            print(f"Loaded projection model from {projection_path}")
        except Exception as e:
            print(f"Error loading projection model: {e}")
            print("Initializing projection model with random weights")
        
        # Load word vectors for complexity scoring
        try:
            print("Loading word embeddings for complexity scoring...")
            self.word_vectors = api.load("glove-wiki-gigaword-100")
            print("Word embeddings loaded successfully")
        except Exception as e:
            print(f"Error loading word vectors: {e}")
            self.word_vectors = None
        
        # Cache for word complexity scores
        self.word_cache = {}

    def compute_word_complexity(self, word):
        """
        Compute complexity score for a word using both tokenization and embedding similarity.
        
        Returns:
            float: Complexity score from 0.0 (simple) to 1.0 (complex)
        """
        clean_word = word.strip(string.punctuation).lower()
        
        # Check cache
        if clean_word in self.word_cache:
            return self.word_cache[clean_word]
        
        # 1. Compute tokenization complexity
        tokens = self.tokenizer.tokenize(clean_word)
        if len(tokens) == 1:
            token_score = 0.0  # Simple word
        elif len(tokens) == 2:
            token_score = 0.3  # Moderate complexity
        elif len(tokens) == 3:
            token_score = 0.6  # Higher complexity
        else:
            token_score = 0.8  # Very complex
        
        # 2. Compute embedding similarity if word vectors available
        embedding_score = 0.0
        if self.word_vectors is not None and len(clean_word) > 2:
            try:
                if clean_word in self.word_vectors:
                    # Find similar words
                    similar_words = self.word_vectors.most_similar(clean_word, topn=5)
                    similarity_scores = [score for _, score in similar_words]
                    avg_similarity = sum(similarity_scores) / len(similarity_scores)
                    
                    # Convert similarity to complexity (higher similarity = lower complexity)
                    embedding_score = 1.0 - avg_similarity
                else:
                    # Not in vocabulary
                    embedding_score = 0.9
            except Exception:
                embedding_score = 0.5
        
        # Combine scores
        if self.word_vectors is not None:
            final_score = 0.6 * token_score + 0.4 * embedding_score
        else:
            final_score = token_score
        
        # Cache result
        self.word_cache[clean_word] = final_score
        return final_score

    def mark_complex_words(self, text):
        """
        Mark complex words with [MASK] based on complexity scoring.
        """
        words = text.split()
        new_words = []
        
        for word in words:
            complexity = self.compute_word_complexity(word)
            if complexity > self.complexity_threshold:
                new_words.append("[MASK]")
            else:
                new_words.append(word)
        
        return " ".join(new_words)

    def split_into_chunks(self, text):
        """
        Splits text into chunks that fit within token limits.
        """
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            tentative = " ".join(current_chunk + [word])
            token_count = len(self.tokenizer.tokenize(tentative))
            
            if token_count <= self.max_chunk_size:
                current_chunk.append(word)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _get_clip_embedding(self, text):
        """
        Get CLIP text embeddings for a chunk of text.
        
        Args:
            text (str): Input text chunk
            
        Returns:
            torch.Tensor: CLIP text embedding vector
        """
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_chunk_size
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(
                input_ids=tokens["input_ids"], 
                attention_mask=tokens["attention_mask"]
            )
        
        return text_features[0]  # Return the embedding for the first (only) item in batch
    
    def get_embeddings(self, text, return_token_level=True):
        """Get text embeddings from CLIP model"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=77
        )
        
        # Get input IDs and attention mask to device
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        with torch.no_grad():
            # Use only the text encoder part of CLIP
            outputs = self.model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get pooled output for sentence-level
            pooled_output = self.model.text_projection(outputs[1])
            
        if return_token_level:
            # Process token-level embeddings from the last hidden state
            token_embeddings = outputs[0][0]  # First batch, all tokens
            
            # Filter out padding, start and end tokens
            masked_embeddings = []
            
            for i, token_id in enumerate(input_ids[0].tolist()):
                # Skip padding (0), [CLS] (49406), [SEP] (49407), etc.
                if token_id > 0 and i > 0 and i < len(token_embeddings)-1 and attention_mask[0][i] == 1:
                    # Project token embeddings to match global embedding space
                    token_embedding = self.model.text_projection(token_embeddings[i].unsqueeze(0)).squeeze(0)
                    masked_embeddings.append(token_embedding)
            
            if not masked_embeddings:
                # If no valid tokens, return the sentence embedding
                return [pooled_output[0]]
                
            return masked_embeddings
        else:
            # Return sentence-level embedding
            return [pooled_output[0]]
    
    

    def aggregate_text(self, text):
        """
        Aggregate text embeddings with improved attention.
        """
        # First, get token-level embeddings
        embeddings = self.get_embeddings(text, return_token_level=False)  # Keep using sentence-level for now
        
        if isinstance(embeddings, list):
            if len(embeddings) > 0:
                # If we got a list of token embeddings, average them
                embeddings = torch.stack(embeddings).mean(dim=0)
            else:
                # Empty list, create a zero tensor
                embeddings = torch.zeros(self.model.text_projection.out_features, device=self.device)
        
        return embeddings