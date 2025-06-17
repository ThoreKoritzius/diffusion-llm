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

# Initialize tokenizer. Ensure a mask token exists.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
if tokenizer.sep_token is None:
    tokenizer.add_special_tokens({"sep_token": "[SEP]"})

MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
SEP_TOKEN_ID = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
print(SEP_TOKEN_ID)
VOCAB_SIZE = len(tokenizer)

PROMPT_LEN = 32   # Conditioning prompt length.
RESP_LEN = 32     # Response length.
TOTAL_SEQ_LEN = PROMPT_LEN + RESP_LEN  # Combined sequence length.

def corrupt_combined(token_ids, noise_level, mask_prompt=True):
    """
    Given token_ids of shape [B, TOTAL_SEQ_LEN],
    if mask_prompt is True, randomly replace each token with MASK_TOKEN_ID with probability noise_level.
    Otherwise, leave the first PROMPT_LEN tokens intact and corrupt only the response tokens.
    Returns:
      - corrupted tensor (same shape)
      - corruption_mask: boolean tensor indicating which positions were replaced.
    """
    B, L = token_ids.size()
    corruption_mask = torch.zeros_like(token_ids).bool()
    if mask_prompt:
        # Entire sequence is eligible.
        corruption_mask = torch.bernoulli(torch.full(token_ids.shape, noise_level, device=token_ids.device, dtype=torch.float32)).bool()
    else:
        # Only the response tokens (positions PROMPT_LEN:) are eligible.
        response_mask = torch.bernoulli(torch.full((B, RESP_LEN), noise_level, device=token_ids.device, dtype=torch.float32)).bool()
        corruption_mask[:, PROMPT_LEN:] = response_mask
    corrupted = token_ids.clone()
    corrupted[corruption_mask] = MASK_TOKEN_ID
    return corrupted, corruption_mask

class MaskedDiffusionLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
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
        
    def forward(self, prompt_ids, response_ids, noise_level, mask_prompt=True):
        """
        Combined forward pass for pretraining or fine-tuning.
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
        B = prompt_ids.size(0)
        # Concatenate prompt and response.
        combined_ids = torch.cat([prompt_ids, response_ids], dim=1)  # [B, TOTAL_SEQ_LEN]
        # Step 1: Corrupt tokens.
        corrupted_ids, corruption_mask = corrupt_combined(combined_ids, noise_level, mask_prompt)
        cids = np.array(corrupted_ids.numpy()[0])
        # Step 2: Embed tokens and add positional encoding.
        x = self.token_embedding(corrupted_ids)  # [B, TOTAL_SEQ_LEN, D]
        x = x + self.pos_embedding[:, :TOTAL_SEQ_LEN, :]
        # Step 3: Process through transformer.
        x = self.transformer(x.transpose(0,1))  # [TOTAL_SEQ_LEN, B, D]
        x = x.transpose(0,1)  # [B, TOTAL_SEQ_LEN, D]
        x = self.norm(x)
        
        # Step 4: Project to logits.
        logits = self.vocab_decoder(x)  # [B, TOTAL_SEQ_LEN, vocab_size]
        
        return logits, corruption_mask

def train(model, tokenizer, dataset, num_steps=1000, mask_prompt=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Adding a learning rate scheduler with decay every 50 steps.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
    model.train()
    for step in range(num_steps):
        # Cycle through dataset.
        prompt_text, response_text = dataset[step % len(dataset)]
        prompt_inputs = tokenizer(prompt_text + "[SEP]", return_tensors="pt", padding="max_length", 
                                  max_length=PROMPT_LEN, truncation=True)
        response_inputs = tokenizer(response_text, return_tensors="pt", padding="max_length", 
                                    max_length=RESP_LEN, truncation=True)
        prompt_ids = prompt_inputs.input_ids
        response_ids = response_inputs.input_ids
        noise_level = random.uniform(0.1, 1)
        
        logits, corruption_mask = model(prompt_ids, response_ids, noise_level, mask_prompt)
        loss = F.cross_entropy(logits[corruption_mask], torch.cat([prompt_ids, response_ids], dim=1)[corruption_mask])
        
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping to keep gradients stable.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        if step % 10 == 0:
            stage = "Pretraining" if mask_prompt else "Fine-tuning"
            current_lr = scheduler.get_last_lr()[0]
            print(f"[{stage}] Step {step} - Loss: {loss.item():.4f} - LR: {current_lr:.6f}")
            

def inference(model, tokenizer, prompt_text):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # Tokenize the prompt.
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding="max_length",
                                  max_length=PROMPT_LEN, truncation=True)
        prompt_ids = prompt_inputs.input_ids.to(device)  # [B, PROMPT_LEN]
        B = prompt_ids.size(0)
        # Initialize all response tokens as MASK.
        response_ids = torch.full((B, RESP_LEN), MASK_TOKEN_ID, device=device)
        
        print("---- Inference: Diffusion Inference ----")
        print("PROMPT", prompt_text)
        # Loop until all tokens are locked or we reach RESP_LEN steps.
        for step in range(RESP_LEN):
            # Forward pass. noise_level=0 since we use predictions directly.
            logits, _ = model(prompt_ids, response_ids, noise_level=0, mask_prompt=False)
            logits_resp = logits[:, PROMPT_LEN:, :]  # shape: [B, RESP_LEN, vocab_size]
            
            # Convert logits to probabilities.
            probs = torch.softmax(logits_resp, dim=-1)  # [B, RESP_LEN, vocab_size]
            top_probs, top_ids = probs.max(dim=-1)        # [B, RESP_LEN]
            
            # Get positions that are still MASK.
            masked_positions = response_ids.eq(MASK_TOKEN_ID)  # [B, RESP_LEN]
            
            # For each example in the batch, lock in one token (the highest confidence among masked ones).
            for b in range(B):
                # Find indices of response tokens still masked.
                candidates = torch.nonzero(masked_positions[b]).squeeze(1)
                if candidates.numel() == 0:
                    continue  # All tokens are locked.
                # Select the candidate with highest confidence.
                candidate_confidences = top_probs[b, candidates]
                best_idx = torch.argmax(candidate_confidences).item()
                best_position = candidates[best_idx].item()
                # Lock in the token at that position.
                response_ids[b, best_position] = top_ids[b, best_position]
            
            # Decode without skipping special tokens.
            decoded = tokenizer.decode(response_ids[0], skip_special_tokens=False)
            # Remove other undesired special tokens (e.g., eos_token).
            if tokenizer.eos_token is not None:
                decoded = decoded.replace(tokenizer.eos_token, "")
            # Replace mask token with a gray-colored version.
            decoded = decoded.replace(tokenizer.mask_token, "\033[90m" + tokenizer.mask_token + "\033[0m")
            print(f"Inference Step {step+1:02d}: {decoded}")
            
            # Stop early if no masked tokens remain.
            if masked_positions.sum() == 0:
                break
    return decoded

if __name__ == "__main__":
    # Define a dataset with (prompt, response) pairs.
    dataset = [
        ("Count the rows of cars", "SELECT COUNT(*) FROM cars"),
        ("Give all cars", "SELECT * FROM cars"),
        ("Give all cars costing over 10k", "SELECT * FROM cars WHERE price > 10000"),
        ("List all cars made after 2015", "SELECT * FROM cars WHERE year > 2015"),
        ("List all red cars", "SELECT * FROM cars WHERE color = 'red'"),
        ("Show all automatic cars", "SELECT * FROM cars WHERE transmission = 'automatic'"),
        ("List all diesel cars", "SELECT * FROM cars WHERE fuel_type = 'diesel'"),
        ("List all cars under $5,000", "SELECT * FROM cars WHERE price < 5000"),
        ("Count of cars per brand", "SELECT brand, COUNT(*) FROM cars GROUP BY brand"),
        ("Average price per brand", "SELECT brand, AVG(price) FROM cars GROUP BY brand"),
        ("Find most expensive car", "SELECT * FROM cars ORDER BY price DESC LIMIT 1"),
        ("Find cheapest car", "SELECT * FROM cars ORDER BY price ASC LIMIT 1"),
        ("List top 10 cheapest cars", "SELECT * FROM cars ORDER BY price ASC LIMIT 10"),
        ("List top 5 newest cars", "SELECT * FROM cars ORDER BY year DESC LIMIT 5"),
        ("Get all BMW cars", "SELECT * FROM cars WHERE brand = 'BMW'"),
        ("Get all cars with mileage under 50k", "SELECT * FROM cars WHERE mileage < 50000"),
        ("List cars between $5k and $15k", "SELECT * FROM cars WHERE price BETWEEN 5000 AND 15000"),
        ("List cars made between 2010 and 2020", "SELECT * FROM cars WHERE year BETWEEN 2010 AND 2020"),
        ("Show cars sorted by price descending", "SELECT * FROM cars ORDER BY price DESC"),
        ("Show cars sorted by mileage ascending", "SELECT * FROM cars ORDER BY mileage ASC"),
        ("How many electric cars?", "SELECT COUNT(*) FROM cars WHERE fuel_type = 'electric'"),
        ("List all manual cars", "SELECT * FROM cars WHERE transmission = 'manual'"),
        ("List cars in black or white", "SELECT * FROM cars WHERE color IN ('black', 'white')"),
        ("Cars not red", "SELECT * FROM cars WHERE color != 'red'"),
        ("Cars with 'Toyota' in brand", "SELECT * FROM cars WHERE brand LIKE '%Toyota%'"),
        ("Cars with model containing 'Civic'", "SELECT * FROM cars WHERE model LIKE '%Civic%'"),
        ("List unique fuel types", "SELECT DISTINCT fuel_type FROM cars"),
        ("Average mileage of diesel cars", "SELECT AVG(mileage) FROM cars WHERE fuel_type = 'diesel'"),
        ("Max mileage for each brand", "SELECT brand, MAX(mileage) FROM cars GROUP BY brand"),
        ("Cars grouped by transmission", "SELECT transmission, COUNT(*) FROM cars GROUP BY transmission"),
        ("List all dealers", "SELECT * FROM dealers"),
        ("Cars sold by dealer ID 3", "SELECT * FROM cars WHERE dealer_id = 3"),
        ("List all cars from dealer 'AutoMart'", "SELECT * FROM cars JOIN dealers ON cars.dealer_id = dealers.dealer_id WHERE dealers.name = 'AutoMart'"),
        ("Show average car price per dealer", "SELECT dealer_id, AVG(price) FROM cars GROUP BY dealer_id"),
        ("Cars newer than 2020 with price below $20k", "SELECT * FROM cars WHERE year > 2020 AND price < 20000"),
        ("List cars with price per year", "SELECT year, AVG(price) FROM cars GROUP BY year"),
        ("Get number of cars per color", "SELECT color, COUNT(*) FROM cars GROUP BY color"),
        ("Count cars with mileage above 100k", "SELECT COUNT(*) FROM cars WHERE mileage > 100000"),
        ("List cars not from USA", "SELECT * FROM cars JOIN brands ON cars.brand = brands.name WHERE brands.country != 'USA'"),
        ("Cars with unknown mileage", "SELECT * FROM cars WHERE mileage IS NULL"),
        ("List cars with a known mileage", "SELECT * FROM cars WHERE mileage IS NOT NULL"),
        ("Get total car inventory value", "SELECT SUM(price) FROM cars"),
        ("Find oldest car", "SELECT * FROM cars ORDER BY year ASC LIMIT 1"),
        ("Find average age of cars", "SELECT AVG(2025 - year) FROM cars"),
        ("Get all cars with model starting with 'A'", "SELECT * FROM cars WHERE model LIKE 'A%'"),
        ("List cars by model alphabetically", "SELECT * FROM cars ORDER BY model ASC"),
        ("Show total cars per year", "SELECT year, COUNT(*) FROM cars GROUP BY year ORDER BY year DESC"),
        ("List green cars under 15k", "SELECT * FROM cars WHERE color = 'green' AND price < 15000"),
        ("List electric cars under 10k", "SELECT * FROM cars WHERE fuel_type = 'electric' AND price < 10000"),
        ("Cars priced above brand average", "SELECT * FROM cars c WHERE price > (SELECT AVG(price) FROM cars WHERE brand = c.brand)"),
    ]

    # Initialize the masked diffusion LM.
    embed_dim = 128
    num_layers = 2
    num_heads = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedDiffusionLM(VOCAB_SIZE, embed_dim, num_layers, num_heads).to(device)
    
    print("====== Pretraining ======")
    # For pretraining, mask_prompt=True (random masking over entire sequence).
    train(model, tokenizer, dataset, num_steps=400, mask_prompt=True)
    torch.save(model.state_dict(), "model_pretrained.pth")
    print("Model saved as model_pretrained.pth")
    
    print("\n====== Inference ======")
    # During inference, we keep the prompt intact and update only the response.
    inference(model, tokenizer, "How many cars are there?")
    inference(model, tokenizer, "Count all cars")
    inference(model, tokenizer, "Count all cars costing more than 20k")