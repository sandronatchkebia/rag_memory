"""Manages text embeddings for semantic search."""

from typing import List, Union, Optional
import numpy as np
import os
import logging
import re

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages generation and storage of text embeddings."""
    
    def __init__(self, model_name: str = None, provider: str = None):
        # Determine provider and model
        self.provider = provider or os.getenv("EMBEDDING_PROVIDER", "openai")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        
        # Initialize based on provider
        if self.provider == "openai":
            self._openai_client = None
            self._local_model = None
        else:
            self._openai_client = None
            self._local_model = None
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize the embedding model."""
        try:
            if self.provider == "openai":
                await self._initialize_openai()
            else:
                await self._initialize_local()
            
            self._initialized = True
            logger.info(f"Embedding manager initialized with {self.provider}:{self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding manager: {e}")
            raise
    
    async def _initialize_openai(self):
        """Initialize OpenAI embeddings."""
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self._openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {e}")
    
    async def _initialize_local(self):
        """Initialize local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading local embedding model: {self.model_name}")
            self._local_model = SentenceTransformer(self.model_name)
            logger.info("Local embedding model loaded successfully")
            
        except ImportError:
            raise ImportError("sentence-transformers package not installed. Run: pip install sentence-transformers")
        except Exception as e:
            raise Exception(f"Failed to load local embedding model: {e}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string."""
        if not self._initialized:
            await self.initialize()
        
        if not text:
            return [0.0] * self.embedding_dimension
        
        try:
            cleaned_text = self._clean_text(text)
            
            if self.provider == "openai":
                return await self._generate_openai_embedding(cleaned_text)
            else:
                return await self._generate_local_embedding(cleaned_text)
                
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            return [0.0] * self.embedding_dimension
    
    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            response = self._openai_client.embeddings.create(
                model=self.model_name,
                input=text
            )
            
            # Extract embedding from response
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    async def _generate_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using local model."""
        try:
            embedding = self._local_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self._initialized:
            await self.initialize()
        
        if not texts:
            return []
        
        try:
            # Clean and normalize texts
            cleaned_texts = [self._clean_text(text) for text in texts]
            
            # Filter out empty texts and ensure we have the right number
            valid_texts = [(i, text) for i, text in enumerate(cleaned_texts) if text and text.strip() and text != "[empty message]" and text != "[error processing message]"]
            
            if not valid_texts:
                # Return zero vectors for all texts
                dim = self.embedding_dimension
                return [[0.0] * dim for _ in texts]
            
            if self.provider == "openai":
                return await self._generate_openai_embeddings_batch(valid_texts, texts)
            else:
                return await self._generate_local_embeddings_batch(valid_texts, texts)
                
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Return zero vectors on error to maintain array lengths
            dim = self.embedding_dimension
            return [[0.0] * dim for _ in texts]
    
    async def _generate_openai_embeddings_batch(self, valid_texts: List[tuple], all_texts: List[str]) -> List[List[float]]:
        """Generate OpenAI embeddings in batch with input sanitization and sub-batching.

        This method enforces conservative limits to avoid API errors:
        - Max inputs per request: 1000 (well under typical limits)
        - Approx token budget per request: 280000 (under 300k)
        - Each string truncated to ~8000 tokens (≈ 24k chars)
        """
        try:
            indices, valid_texts_only = zip(*valid_texts)

            # Sanitize and optionally chunk overly long texts
            sanitized_texts: List[str] = []
            sanitized_indices: List[int] = []
            for idx, text in zip(indices, valid_texts_only):
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""
                text = text.strip()
                if not text:
                    continue
                # Truncate extremely long strings first (hard cap)
                if len(text) > 24000:
                    chunks = self._chunk_long_text(text)
                    if chunks:
                        text = chunks[0]
                    else:
                        continue
                sanitized_texts.append(text)
                sanitized_indices.append(idx)

            if not sanitized_texts:
                dim = self.embedding_dimension
                return [[0.0] * dim for _ in all_texts]

            # Prepare result holder
            dim = self.embedding_dimension
            result = [[0.0] * dim for _ in all_texts]

            # Sub-batch by item count and approximate token budget
            max_items = 1000
            max_tokens_budget = 280000

            start = 0
            while start < len(sanitized_texts):
                # Determine end index by items first
                end = min(start + max_items, len(sanitized_texts))

                # Further shrink to fit token budget (approx 3 chars per token)
                token_budget_chars = max_tokens_budget * 3
                current_chars = 0
                final_end = start
                for j in range(start, end):
                    length = len(sanitized_texts[j])
                    if current_chars + length > token_budget_chars:
                        break
                    current_chars += length
                    final_end = j + 1

                batch_inputs = sanitized_texts[start:final_end]
                batch_indices = sanitized_indices[start:final_end]

                if not batch_inputs:
                    # Safety fallback to avoid infinite loops
                    start = final_end if final_end > start else start + 1
                    continue

                # Attempt to embed this sub-batch; on API error, bisect to isolate bad inputs
                await self._embed_with_bisect(batch_inputs, batch_indices, result)

                start = final_end

            return result

        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            dim = self.embedding_dimension
            return [[0.0] * dim for _ in all_texts]

    async def _embed_with_bisect(self, inputs: List[str], input_indices: List[int], out_result: List[List[float]]):
        """Embed a list of strings; on API error, bisect to isolate problematic inputs.

        - inputs: list of sanitized strings to embed
        - input_indices: mapping of inputs to positions in out_result
        - out_result: preallocated output matrix to write embeddings into
        """
        if not inputs:
            return
        try:
            response = self._openai_client.embeddings.create(
                model=self.model_name,
                input=inputs
            )
            for out_idx, embedding_data in enumerate(response.data):
                original_idx = input_indices[out_idx]
                if 0 <= original_idx < len(out_result):
                    out_result[original_idx] = embedding_data.embedding
            return
        except Exception as e:
            # If single item fails, skip it with zero vector and continue
            if len(inputs) == 1:
                # Log identifying info: length and sha1 of content (not content itself)
                try:
                    import hashlib
                    text_len = len(inputs[0]) if isinstance(inputs[0], str) else 0
                    text_hash = hashlib.sha1(inputs[0].encode('utf-8', errors='ignore')).hexdigest()[:12] if isinstance(inputs[0], str) else ""
                    logger.warning(f"Embedding skip: index={input_indices[0]}, len={text_len}, sha1={text_hash}, error={e}")
                except Exception:
                    logger.warning(f"Embedding skip: index={input_indices[0]}, error={e}")
                # Leave zero vector in place
                return
            # Otherwise, split and try halves
            mid = len(inputs) // 2
            await self._embed_with_bisect(inputs[:mid], input_indices[:mid], out_result)
            await self._embed_with_bisect(inputs[mid:], input_indices[mid:], out_result)
    
    async def _generate_local_embeddings_batch(self, valid_texts: List[tuple], all_texts: List[str]) -> List[List[float]]:
        """Generate local embeddings in batch."""
        try:
            indices, valid_texts_only = zip(*valid_texts)
            
            embeddings = self._local_model.encode(valid_texts_only, convert_to_tensor=False)
            
            # Create result list with zero vectors for empty texts
            dim = self.embedding_dimension
            result = [[0.0] * dim for _ in all_texts]
            
            # Fill in valid embeddings
            for idx, embedding in zip(indices, embeddings):
                result[idx] = embedding.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Local batch embedding error: {e}")
            raise
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
        
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1, dtype=np.float32)
            vec2 = np.array(embedding2, dtype=np.float32)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def batch_similarity(self, query_embedding: List[float], embeddings: List[List[float]]) -> List[float]:
        """Calculate similarity between query and multiple embeddings."""
        if not query_embedding or not embeddings:
            return []
        
        try:
            query_vec = np.array(query_embedding, dtype=np.float32)
            embedding_matrix = np.array(embeddings, dtype=np.float32)
            
            # Calculate cosine similarities
            similarities = []
            for embedding in embedding_matrix:
                dot_product = np.dot(query_vec, embedding)
                norm1 = np.linalg.norm(query_vec)
                norm2 = np.linalg.norm(embedding)
                
                if norm1 == 0 or norm2 == 0:
                    similarities.append(0.0)
                else:
                    similarity = dot_product / (norm1 * norm2)
                    similarities.append(float(similarity))
            
            return similarities
        except Exception as e:
            logger.error(f"Error calculating batch similarity: {e}")
            return [0.0] * len(embeddings)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for embedding generation."""
        if not text:
            return ""
        
        try:
            # Basic cleaning
            text = text.strip()
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove or replace problematic characters that OpenAI might reject
            # Remove control characters except newlines and tabs
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            
            # Ensure text is valid UTF-8
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Truncate if too long (model has limits)
            if self.provider == "openai":
                # OpenAI ada-002 has a limit of ~8192 tokens
                # Rough estimate: 1 token ≈ 4 characters for English, 1 token ≈ 2 characters for other languages
                # Use conservative estimate of 3 characters per token
                max_tokens = 8000
                max_chars = max_tokens * 3
            else:
                max_chars = 512   # Local model limit
                
            if len(text) > max_chars:
                # Truncate and add indicator
                text = text[:max_chars] + " [truncated]"
            
            # Final safety check - ensure we have valid text
            if not text or text.isspace():
                return "[empty message]"
                
            return text
            
        except Exception as e:
            logger.warning(f"Error cleaning text: {e}, returning safe fallback")
            return "[error processing message]"
    
    def _chunk_long_text(self, text: str, max_tokens: int = 8000) -> List[str]:
        """Split extremely long text into chunks that fit within token limits."""
        if not text:
            return []
        
        # Rough estimate: 1 token ≈ 3 characters
        max_chars = max_tokens * 3
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        # Split by sentences or paragraphs to maintain context
        sentences = text.split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed the limit
            if current_length + len(sentence) + 2 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_length = len(sentence)
                else:
                    # Single sentence is too long, split it
                    if len(sentence) > max_chars:
                        # Split by words
                        words = sentence.split()
                        temp_chunk = ""
                        temp_length = 0
                        
                        for word in words:
                            if temp_length + len(word) + 1 > max_chars:
                                if temp_chunk:
                                    chunks.append(temp_chunk.strip())
                                temp_chunk = word
                                temp_length = len(word)
                            else:
                                temp_chunk += " " + word if temp_chunk else word
                                temp_length += len(word) + 1
                        
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                    else:
                        chunks.append(sentence)
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
                current_length += len(sentence) + 2
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings generated by this model."""
        if self.provider == "openai":
            if "ada-002" in self.model_name:
                return 1536
            elif "3-small" in self.model_name:
                return 1536
            elif "3-large" in self.model_name:
                return 3072
            else:
                return 1536  # Default for OpenAI models
        else:
            if self._local_model:
                return self._local_model.get_sentence_embedding_dimension()
            return 384  # Default for local models
    
    @property
    def is_ready(self) -> bool:
        """Check if the embedding manager is ready to use."""
        return self._initialized and (
            (self.provider == "openai" and self._openai_client is not None) or
            (self.provider == "local" and self._local_model is not None)
        )
    
    @property
    def provider_info(self) -> str:
        """Get information about the current provider."""
        return f"{self.provider}:{self.model_name} (dim: {self.embedding_dimension})"
