import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

# Set all seeds for reproducibility.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add a mask token if not already in the tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

if tokenizer.mask_token is None:
    # We arbitrarily choose "[MASK]" – make sure to resize the model embeddings accordingly.
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
VOCAB_SIZE = len(tokenizer)

# Fixed sequence lengths.
MAX_SEQ_LEN = 32
COND_SEQ_LEN = 32  # for conditioning prompt

def corrupt_tokens(token_ids, noise_level):
    """
    Given a tensor token_ids of shape [B, L],
    randomly replace each token with MASK_TOKEN_ID with probability `noise_level`.
    Returns the corrupted tensor (of same shape) and a mask indicating which positions were corrupted.
    """
    B, L = token_ids.size()
    # Create a corruption mask: 1 for corrupted, 0 for intact.
    corruption_mask = torch.bernoulli(torch.full(token_ids.shape, noise_level, device=token_ids.device)).bool()
    corrupted = token_ids.clone()
    # Replace with MASK_TOKEN_ID where corrupted.
    corrupted[corruption_mask] = MASK_TOKEN_ID
    return corrupted, corruption_mask

class MaskedDiffusionLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super().__init__()
        # Embedding and positional encoding.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, embed_dim))
        # Transformer for denoising (predicting the original token for masked positions).
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Conditioning encoder: We use a second transformer to encode the conditioning prompt.
        cond_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.cond_encoder = nn.TransformerEncoder(cond_layer, num_layers=1)
        # Final projection to vocabulary logits.
        self.vocab_decoder = nn.Linear(embed_dim, vocab_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def encode_condition(self, cond_ids):
        """
        cond_ids: tensor of shape [B, L_cond]
        Returns: condition encoding of shape [L_cond, B, D]
        """
        cond_emb = self.token_embedding(cond_ids)  # [B, L_cond, D]
        cond_emb = cond_emb + self.pos_embedding[:, :cond_ids.size(1), :]
        cond_enc = self.cond_encoder(cond_emb.transpose(0,1))  # [L_cond, B, D]
        return cond_enc

    def forward(self, cond_ids, target_ids, noise_level):
        """
        Training forward pass.
          cond_ids: [B, L_cond] conditioning prompt tokens.
          target_ids: [B, L] target SQL tokens.
          noise_level: a scalar in [0,1] indicating the fraction of tokens to mask.
        Process:
          1. Corrupt target_ids by replacing tokens with MASK_TOKEN_ID with probability noise_level.
          2. Embed the corrupted tokens + fixed positional embedding.
          3. Obtain condition encoding.
          4. Feed the corrupted embeddings to the transformer, optionally attending to the condition.
          5. Predict the original token logits only on positions that were masked.
        Returns:
          logits: [B, L, vocab_size] predicted token logits (for all positions).
          corruption_mask: [B, L] indicator mask for positions that were corrupted.
        """
        B, L = target_ids.size()
        # Step 1: Corrupt the target tokens.
        corrupted_ids, corruption_mask = corrupt_tokens(target_ids, noise_level)
        
        # Step 2: Embed corrupted tokens and add positional encoding.
        corrupted_emb = self.token_embedding(corrupted_ids)  # [B, L, D]
        corrupted_emb = corrupted_emb + self.pos_embedding[:, :L, :]
        
        # Step 3: Encode conditioning prompt.
        cond_enc = self.encode_condition(cond_ids)  # [L_cond, B, D]
        
        # Step 4: Process corrupted embeddings with transformer.
        # Transpose to [L, B, D].
        x = self.transformer(corrupted_emb.transpose(0,1))  # [L, B, D]
        x = x.transpose(0,1)  # [B, L, D]
        x = self.norm(x)
        
        # Optionally, one could add cross-attention with cond_enc here to strengthen conditioning.
        # For simplicity, we concatenate the condition representation (pooled) with each token embedding.
        cond_pooled = cond_enc.mean(dim=0).unsqueeze(1)  # [B, 1, D]
        x = x + cond_pooled  # Broadcasting over L.
        
        # Step 5: Project to vocabulary logits.
        logits = self.vocab_decoder(x)  # [B, L, vocab_size]
        return logits, corruption_mask

def train(model, tokenizer, dataset, num_steps=1000):
    """
    Train on a list of (conditioning_prompt, target_sql) pairs.
    Uses cross entropy only at the positions masked (corrupted).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(num_steps):
        cond_text, target_text = dataset[step % len(dataset)]
        # Tokenize conditioning prompt.
        cond_inputs = tokenizer(cond_text, return_tensors="pt", padding="max_length", 
                                  max_length=COND_SEQ_LEN, truncation=True)
        # Tokenize target SQL.
        target_inputs = tokenizer(target_text, return_tensors="pt", padding="max_length", 
                                    max_length=MAX_SEQ_LEN, truncation=True)
        cond_ids = cond_inputs.input_ids  # [B, L_cond]
        target_ids = target_inputs.input_ids  # [B, L]
        # Sample a noise level uniformly from, say, 0.3 to 0.8.
        noise_level = random.uniform(0.3, 0.8)
        
        logits, corruption_mask = model(cond_ids, target_ids, noise_level)
        # Compute loss only for positions that were masked.
        loss = F.cross_entropy(logits[corruption_mask], target_ids[corruption_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"[Training] Step {step} - Loss: {loss.item():.4f}")

def inference(model, tokenizer, cond_text, num_steps=10):
    """
    Inference with discrete diffusion thought process.
    We start with a fully masked sequence and iteratively update a fraction of tokens.
    At each step, we reduce the masking probability.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # Tokenize conditioning prompt.
        cond_inputs = tokenizer(cond_text, return_tensors="pt", padding="max_length", 
                                  max_length=COND_SEQ_LEN, truncation=True)
        cond_ids = cond_inputs.input_ids.to(device)
        
        B = 1
        L = MAX_SEQ_LEN
        # Start with a fully masked sequence.
        current_ids = torch.full((B, L), MASK_TOKEN_ID, device=device)
        
        print("---- Inference: Iterative Denoising (Masked Diffusion) ----")
        for step in range(num_steps):
            # Gradually reduce noise level; e.g., step 0 uses 0.9 masking, final step 0.0.
            noise_level = 0.9 * (1 - step / (num_steps - 1))
            # Get model predictions.
            logits, _ = model(cond_ids, current_ids, noise_level)
            # Obtain predicted token IDs.
            pred_ids = torch.argmax(logits, dim=-1)
            # Update only the positions that are masked.
            # For simplicity, assume we replace all masked positions with predictions.
            mask_positions = (current_ids == MASK_TOKEN_ID)
            current_ids[mask_positions] = pred_ids[mask_positions]
            decoded = tokenizer.decode(current_ids[0], skip_special_tokens=True)
            print(f"Inference Step {step+1:02d}: {decoded}")
    return decoded

if __name__ == "__main__":
    # Define a dataset with more examples.
    dataset = [
        ("Count the rows of cars", "SELECT COUNT(*) FROM cars"),
        ("Give all cars", "SELECT * FROM cars"),
        ("Give all cars costing over 10k", "SELECT * FROM cars WHERE price > 10K")
    ]
    # Initialize the masked diffusion LM.
    embed_dim = 128
    num_layers = 2
    num_heads = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedDiffusionLM(VOCAB_SIZE, embed_dim, num_layers, num_heads).to(device)
    
    print("====== Training ======")
    train(model, tokenizer, dataset, num_steps=700)
    torch.save(model.state_dict(), "model_masked.pth")
    print("Model saved as model_masked.pth")
    
    print("\n====== Inference ======")
    inference(model, tokenizer, "How many cars are there?", num_steps=10)
    inference(model, tokenizer, "Count all cars", num_steps=10)
    inference(model, tokenizer, "Count all cars costing more than 20k", num_steps=10)