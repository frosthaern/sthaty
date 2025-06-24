from __future__ import annotations
from typing import Generator, List, Tuple, Optional
import logging
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
import umap.umap_ as umap_lib
from tqdm.auto import tqdm
from modules.classes.AttentionExtractor import AttentionExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttentionAnalyzer:
    """Analyzes attention patterns from an AttentionExtractor.
    
    This class provides methods to analyze and visualize attention patterns
    from transformer models, particularly focusing on CodeBERT's attention heads.
    """
    
    def __init__(self, extractor: AttentionExtractor, show_progress: bool = True):
        """Initialize the AttentionAnalyzer with an AttentionExtractor instance.
        
        Args:
            extractor: An instance of AttentionExtractor that provides attention data
            show_progress: Whether to display progress bars for long-running operations
        """
        if not isinstance(extractor, AttentionExtractor):
            raise TypeError("extractor must be an instance of AttentionExtractor")
        self.extractor = extractor
        self.show_progress = show_progress
        
    def _get_progress_bar(self, iterable, desc: Optional[str] = None, total: Optional[int] = None, **kwargs):
        """Helper method to get a progress bar or the original iterable based on show_progress.
        
        Args:
            iterable: The iterable to wrap with tqdm
            desc: Description for the progress bar
            total: Total number of items (if not len(iterable))
            **kwargs: Additional arguments to pass to tqdm
            
        Returns:
            Either a tqdm-wrapped iterable or the original iterable
        """
        if not self.show_progress:
            return iterable
            
        if total is None and hasattr(iterable, '__len__'):
            total = len(iterable)
            
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            leave=False,
            **kwargs
        )

    def print_info(self, max_samples: Optional[int] = 10) -> None:
        """Print information about the attention data structure.
        
        This is useful for debugging and understanding the shape and structure
        of the attention tensors.
        
        Args:
            max_samples: Maximum number of samples to display. If None, shows all.
        """
        try:
            samples = []
            for i, attn in enumerate(self.extractor):
                if not isinstance(attn, dict) or "heads" not in attn or "special_ids" not in attn:
                    logger.warning(f"Unexpected attention data format at index {i}")
                    continue
                    
                samples.append(attn)
                if max_samples is not None and len(samples) >= max_samples:
                    break
            
            logger.info(f"Found {len(samples)} samples:")
            for i, attn in enumerate(samples):
                logger.info(f"Sample {i}:")
                logger.info(f"  Heads shape: {attn['heads'].shape}")
                logger.info(f"  Special IDs: {attn['special_ids']}")
                
        except Exception as e:
            logger.error(f"Error printing attention info: {str(e)}")
            raise

    def _validate_attention_window(
        self, 
        qb: int, 
        qe: int, 
        cb: int, 
        ce: int, 
        seq_len: int
    ) -> Tuple[bool, Tuple[int, int, int, int]]:
        """Validate and adjust the attention window indices.
        
        Args:
            qb: Query begin index
            qe: Query end index
            cb: Context begin index
            ce: Context end index
            seq_len: Sequence length
            
        Returns:
            Tuple of (is_valid, (qb, qe, cb, ce)) where is_valid indicates if the window is valid
        """
        try:
            # Adjust indices to be within valid range
            qb = max(0, min(qb + 1, seq_len - 1))
            qe = max(qb, min(qe, seq_len))
            cb = max(0, min(cb + 1, seq_len - 1))
            ce = max(cb, min(ce, seq_len))
            
            # Check if the window is valid
            is_valid = not (qe <= qb or ce <= cb)
            if not is_valid:
                logger.warning(f"Invalid attention window: qb={qb}, qe={qe}, cb={cb}, ce={ce}")
                
            return is_valid, (qb, qe, cb, ce)
            
        except Exception as e:
            logger.error(f"Error validating attention window: {str(e)}")
            return False, (0, 0, 0, 0)

    def entropy(self) -> Generator[np.ndarray, None, None]:
        """Calculate the entropy of attention weights for each head and layer.
        
        The entropy is calculated over the attention weights within the specified
        query-context window for each attention head in each layer.
        
        Yields:
            np.ndarray: A 2D array of shape (layers, heads) containing the mean entropy
                       of attention weights for each head and layer.
                       
        Raises:
            ValueError: If the extractor is not properly initialized or if the attention data is invalid.
        """
        if not hasattr(self, 'extractor') or not self.extractor:
            raise ValueError("No extractor provided. Initialize with a valid AttentionExtractor.")
        
        # Get total number of samples if possible for progress bar
        total = None
        if hasattr(self.extractor, '__len__'):
            total = len(self.extractor)
            
        for attn in self._get_progress_bar(self.extractor, desc="Calculating attention entropy", total=total):
            try:
                # Validate input
                if not isinstance(attn, dict) or "special_ids" not in attn or "heads" not in attn:
                    raise ValueError("Invalid attention data format: missing required keys")
                    
                qb, qe, cb, ce = attn["special_ids"]
                heads_tensor = attn["heads"]
                
                # Handle both 3D and 4D attention tensors
                if heads_tensor.dim() == 3:
                    # For 3D tensor: assume it's a single layer with shape [heads, seq_len, seq_len]
                    heads_tensor = heads_tensor.unsqueeze(0)  # Add layer dimension
                    layers, heads, seq_len, _ = heads_tensor.shape
                elif heads_tensor.dim() == 4:
                    layers, heads, seq_len, _ = heads_tensor.shape
                else:
                    raise ValueError(
                        f"Expected 3D [heads, seq_len, seq_len] or 4D [layers, heads, seq_len, seq_len] attention tensor, "
                        f"got shape {tuple(heads_tensor.shape)} with {heads_tensor.dim()} dimensions"
                    )
                    
                entropy = np.zeros((layers, heads), dtype=np.float64)
                
                # Validate and adjust the attention window
                is_valid, (qb, qe, cb, ce) = self._validate_attention_window(qb, qe, cb, ce, seq_len)
                if not is_valid:
                    yield entropy  # Return zero entropy for invalid window
                    continue
                
                # Process each head and layer with progress bar if there are many
                total_heads = layers * heads
                head_iter = (
                    (layer_idx, head_idx) 
                    for layer_idx in range(layers) 
                    for head_idx in range(heads)
                )
                
                for layer, head in self._get_progress_bar(
                    head_iter, 
                    desc=f"Processing {layers} layers x {heads} heads",
                    total=total_heads,
                    leave=False
                ):
                        try:
                            # Get attention scores for this head and layer
                            head_attention = heads_tensor[layer, head, qb:qe, cb:ce]
                            head_attention_score = head_attention.cpu().numpy()
                            
                            if head_attention_score.size == 0:
                                continue
                            
                            # Calculate entropy with numerical stability
                            row_sum = head_attention_score.sum(axis=1, keepdims=True)
                            normalized_attention = head_attention_score / (row_sum + 1e-10)
                            log_attention = np.log(normalized_attention + 1e-10)
                            token_entropies = -np.sum(normalized_attention * log_attention, axis=1)
                            entropy[layer][head] = np.mean(token_entropies)
                            
                        except Exception as e:
                            logger.error(f"Error processing layer {layer}, head {head}: {str(e)}")
                            entropy[layer][head] = 0.0
                
                yield entropy
                
            except Exception as e:
                logger.error(f"Error processing attention: {str(e)}")
                # Return zero entropy with appropriate shape if possible
                if 'layers' in locals() and 'heads' in locals():
                    yield np.zeros((layers, heads), dtype=np.float64)
                else:
                    # Fallback to a common shape if we can't determine the correct one
                    yield np.zeros((12, 12), dtype=np.float64)

    def _batch_process_vectors(
        self, 
        vectors: List[NDArray[np.float64]], 
        batch_size: int = 1000
    ) -> Generator[NDArray[np.float64], None, None]:
        """Process vectors in batches to reduce memory usage.
        
        Args:
            vectors: List of input vectors to process
            batch_size: Number of vectors to process in each batch
            
        Yields:
            Processed vectors one by one
        """
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            for vector in batch:
                yield vector

    def pca(
        self, 
        vectors: List[NDArray[np.float64]], 
        n_components: int = 30,
        batch_size: int = 1000,
        **pca_kwargs
    ) -> Generator[NDArray[np.float64], None, None]:
        """Apply PCA dimensionality reduction to the input vectors.
        
        Args:
            vectors: List of input vectors to reduce
            n_components: Number of components to keep (recommended <= 50 for visualization)
            batch_size: Number of vectors to process in each batch
            **pca_kwargs: Additional arguments to pass to PCA
            
        Yields:
            Reduced-dimension vectors one by one
            
        Raises:
            ValueError: If input vectors are empty or have inconsistent shapes
        """
        if not vectors:
            logger.warning("No vectors provided to PCA")
            return
            
        try:
            # Ensure all vectors are numpy arrays and have the same shape
            vectors = [np.asarray(v, dtype=np.float64) for v in vectors]
            first_shape = vectors[0].shape
            
            if not all(v.shape == first_shape for v in vectors):
                raise ValueError(f"All input vectors must have the same shape. "
                               f"First shape: {first_shape}, found different shapes in input vectors.")
            
            # Flatten vectors if they're not 1D
            flat_vectors = np.vstack([v.reshape(1, -1) for v in vectors])
            
            # Configure PCA
            n_components = min(n_components, flat_vectors.shape[1])
            pca = PCA(n_components=n_components, **pca_kwargs)
            
            # Fit PCA on a sample if the dataset is large
            sample_size = min(10000, len(vectors))
            with self._get_progress_bar(
                [0],  # Dummy iterable for progress bar
                desc=f"Fitting PCA on {sample_size} samples",
                total=1
            ):
                pca.fit(flat_vectors[:sample_size])
            
            # Transform vectors in batches
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            
            for i in self._get_progress_bar(
                range(0, len(vectors), batch_size),
                desc="Transforming vectors with PCA",
                total=total_batches,
                unit="batch"
            ):
                batch = flat_vectors[i:i + batch_size]
                transformed = pca.transform(batch)
                
                # Yield each vector in the batch
                for j in range(transformed.shape[0]):
                    yield transformed[j]
                    
        except Exception as e:
            logger.error(f"Error in PCA reduction: {str(e)}")
            # Yield zeros with the expected shape if there's an error
            zero_vec = np.zeros(n_components, dtype=np.float64)
            for _ in vectors:
                yield zero_vec

    def umap(
        self, 
        vectors: List[NDArray[np.float64]], 
        n_components: int = 30,
        batch_size: int = 1000,
        **umap_kwargs
    ) -> Generator[NDArray[np.float64], None, None]:
        """Apply UMAP dimensionality reduction to the input vectors.
        
        Args:
            vectors: List of input vectors to reduce
            n_components: Number of components to keep (2 or 3 recommended for visualization)
            batch_size: Number of vectors to process in each batch
            **umap_kwargs: Additional arguments to pass to UMAP
            
        Yields:
            Reduced-dimension vectors one by one
            
        Raises:
            ValueError: If input vectors are empty or have inconsistent shapes
        """
        if not vectors:
            logger.warning("No vectors provided to UMAP")
            return
            
        try:
            # Ensure all vectors are numpy arrays and have the same shape
            vectors = [np.asarray(v, dtype=np.float64) for v in vectors]
            first_shape = vectors[0].shape
            
            if not all(v.shape == first_shape for v in vectors):
                raise ValueError(f"All input vectors must have the same shape. "
                               f"First shape: {first_shape}, found different shapes in input vectors.")
            
            # Flatten vectors if they're not 1D
            flat_vectors = np.vstack([v.reshape(1, -1) for v in vectors])
            
            # Configure UMAP with default parameters unless overridden
            umap_params = {
                'n_components': min(n_components, flat_vectors.shape[1]),
                'n_neighbors': min(15, len(vectors) - 1),
                'min_dist': 0.1,
                'metric': 'euclidean',
                'n_jobs': -1,
                **umap_kwargs
            }
            
            # Initialize UMAP
            umap_reducer = umap_lib.UMAP(**umap_params)
            
            # Fit UMAP on a sample if the dataset is large
            sample_size = min(10000, len(vectors))
            with self._get_progress_bar(
                [0],  # Dummy iterable for progress bar
                desc=f"Fitting UMAP on {sample_size} samples",
                total=1
            ):
                umap_reducer.fit(flat_vectors[:sample_size])
            
            # Transform vectors in batches
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            
            for i in self._get_progress_bar(
                range(0, len(vectors), batch_size),
                desc="Transforming vectors with UMAP",
                total=total_batches,
                unit="batch"
            ):
                batch = flat_vectors[i:i + batch_size]
                transformed = umap_reducer.transform(batch)
                
                # Yield each vector in the batch
                for j in range(transformed.shape[0]):
                    yield transformed[j]
                    
        except Exception as e:
            logger.error(f"Error in UMAP reduction: {str(e)}")
            # Yield zeros with the expected shape if there's an error
            zero_vec = np.zeros(n_components, dtype=np.float64)
            for _ in vectors:
                yield zero_vec
