from typing import Generator, Dict, List, Any, Optional, Union
import torch
from torch import Tensor
from transformers import AutoModel
from tqdm import tqdm

from modules.absclasses.LoadTokenizedDataAbs import LoadTokenizedDataAbs

class AttentionExtractor:
    """Extracts attention heads from CodeBERT model for given inputs."""
    
    def __init__(
        self, 
        encoder: LoadTokenizedDataAbs, 
        batch_size: int = 8,
        device: Optional[Union[str, torch.device]] = None,
        show_progress: bool = True
    ):
        self.encoder = encoder
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = AutoModel.from_pretrained(
            "microsoft/codebert-base", 
            attn_implementation="eager",
            output_attentions=True
        ).to(self.device)
        self.model.eval()
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Tensor]]:
        """Process a batch of encodings through the model."""
        try:
            # Move batch to device
            input_ids = torch.stack([item['input_ids'] for item in batch]).to(self.device)
            attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    return_dict=True
                )
            
            # Process each item in the batch
            results = []
            for i, (attention_heads, special_ids) in enumerate(zip(outputs.attentions, [b['special_ids'] for b in batch])):
                heads_per_layer = [ah[i].cpu() for ah in attention_heads]  # Move to CPU to free GPU memory
                results.append({
                    'heads': torch.stack(heads_per_layer, dim=0),
                    'special_ids': special_ids
                })
            
            return results
            
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return []
    
    def __iter__(self) -> Generator[Dict[str, Tensor], None, None]:
        """Iterate over the dataset, yielding attention heads for each item."""
        batch = []
        encoder_iter = tqdm(self.encoder, desc="Extracting attention") if self.show_progress else self.encoder
        
        for encodings in encoder_iter:
            batch.append(encodings)
            if len(batch) >= self.batch_size:
                for result in self._process_batch(batch):
                    if result:  # Only yield if processing was successful
                        yield result
                batch = []
        
        # Process the last batch if it's not empty
        if batch:
            for result in self._process_batch(batch):
                if result:
                    yield result
      
