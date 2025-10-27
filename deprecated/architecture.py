
import numpy as np
import torch
import torch.nn as nn

class MaskedDiffusionLM(nn.Module):
    def __init__(self, TOTAL_SEQ_LEN, vocab_size, embed_dim, num_layers, num_heads):
        super().__init__()
        # Embedding and positional encoding for the combined sequence.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, TOTAL_SEQ_LEN, embed_dim))
        # A single transformer for the denoising task.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final projection to vocabulary logits.
        self.vocab_decoder = nn.Linear(embed_dim, vocab_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, combined_ids, attention_mask=None):
        """
        Combined forward pass for pretraining or fine-tuning.
          prompt_len:
          prompt_ids: [B, PROMPT_LEN] tokens for prompt.
          response_ids: [B, RESP_LEN] tokens for response.
          noise_level: scalar in [0,1] (masking probability).
          mask_prompt: If True, mask entire sequence (pretraining). If False, only mask the response.
        Process:
          1. Concatenate prompt and response into one [B, TOTAL_SEQ_LEN] sequence.
          2. Apply corruption per the mask strategy.
          3. Embed tokens + add positional embeddings.
          4. Process through the transformer.
          5. Project to vocabulary logits.
        Returns:
          logits: [B, TOTAL_SEQ_LEN, vocab_size]
          corruption_mask: boolean [B, TOTAL_SEQ_LEN] (indicating masked positions).
        """
        # Embed tokens and add positional encoding.
        x = self.token_embedding(combined_ids)  # [B, seq_len, D]
        seq_len = combined_ids.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # If an attention mask is provided, create a key_padding_mask for the transformer.
        key_padding_mask = None
        if attention_mask is not None:
            # Transformer expects True for positions to ignore (i.e., the padded tokens).
            key_padding_mask = ~attention_mask.bool()
        
        # Process through transformer using the key_padding_mask.
        x = self.transformer(x.transpose(0, 1), src_key_padding_mask=key_padding_mask).transpose(0, 1)
        x = self.norm(x)
        logits = self.vocab_decoder(x)  # [B, seq_len, vocab_size]
        return logits
