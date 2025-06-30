import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from dataset import pretrain_dataset, finetune_dataset, inference_dataset
from time import time
from transformers import get_scheduler

def format_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:  # Less than 60 seconds
        return f"{seconds}s"
    
    elif seconds < 3600:  # Less than 1 hour (in minutes and seconds)
        minutes = seconds // 60
        seconds %= 60
        return f"{minutes}m {seconds}s"
    
    else:  # Greater than or equal to 1 hour (in hours and minutes)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"

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

LOCKING = True
TOTAL_SEQ_LEN = 64

def corrupt_combined(token_ids, noise_level, mask_prompt=True, prompt_length=None):
    """
    Given token_ids of shape [B, L],
    if mask_prompt is True, randomly replace each token with MASK_TOKEN_ID with probability noise_level.
    Otherwise, leave the first prompt_length tokens intact and corrupt only the response tokens.
    Returns:
      - corrupted tensor (same shape)
      - corruption_mask: boolean tensor indicating which positions were replaced.
    """
    B, L = token_ids.size()
    corruption_mask = torch.zeros_like(token_ids, device=token_ids.device).bool()
    if mask_prompt:
        # Entire sequence is eligible.
        corruption_mask = torch.bernoulli(
            torch.full(token_ids.shape, noise_level, device=token_ids.device, dtype=torch.float32)
        ).bool()
    else:
        # Only the response tokens (positions prompt_length:) are eligible.
        if prompt_length is None:
            raise ValueError("prompt_length must be provided when mask_prompt is False.")
        response_len = L - prompt_length
        response_mask = torch.bernoulli(
            torch.full((B, response_len), noise_level, device=token_ids.device, dtype=torch.float32)
        ).bool()
        corruption_mask[:, prompt_length:] = response_mask
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

def train(model, tokenizer, dataset, num_steps=1000, mask_prompt=True, accumulation_steps=1, warmup_steps=500):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)
    
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_steps
    )

    model.train()
    start = time()
    total_loss = 0.0
    device = next(model.parameters()).device
    
    for step in range(num_steps):
        if step % accumulation_steps == 0:
            optimizer.zero_grad()
        
        prompt_text, response_text = dataset[step % len(dataset)]

        prompt_inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                                  max_length=(TOTAL_SEQ_LEN -2))
        prompt_ids = prompt_inputs.input_ids.to(device)
        response_inputs = tokenizer(response_text, return_tensors="pt", padding="max_length",
                                    max_length=TOTAL_SEQ_LEN - prompt_ids.size(1), truncation=True)

        response_ids = response_inputs.input_ids.to(device)
        prompt_attention_mask = prompt_inputs.attention_mask.to(device)
        response_attention_mask = response_inputs.attention_mask.to(device)
        combined_ids = torch.cat([prompt_ids, response_ids], dim=1)  # [B, total_seq_len]

        noise_level = random.uniform(0.1, 0.9)
        corrupted_ids, corruption_mask = corrupt_combined(combined_ids, noise_level, mask_prompt,
                                                          prompt_length=prompt_ids.size(1))

        logits = model(corrupted_ids)

        # Combine ids and attention mask
        combined_attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1).bool()

        # Valid positions: both masked by corruption and not padding
        valid_mask = corruption_mask & combined_attention_mask
        if valid_mask.sum() == 0:
            continue  # Skip if nothing to supervise

        loss = F.cross_entropy(logits[valid_mask], combined_ids[valid_mask])
        loss = loss / accumulation_steps
        loss.backward()
        total_loss += loss.item()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
        
        if step % 100 == 0:
            avg_loss = total_loss / (step + 1)
            current_lr = lr_scheduler.get_last_lr()[0]
            percent = 100.0 * step / max(1, num_steps)
            elapsed = time() - start
            print(f"[{'Pretraining' if mask_prompt else 'Fine-tuning'}] Step {step}/{num_steps} "
                  f"({percent:.1f}%, {format_time(elapsed)}/{format_time(elapsed / max(percent / 100, 0.01))}) "
                  f"- Avg Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")
    
    print(f"Training complete in {format_time(time() - start)}")

def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    for b in range(logits.size(0)):
        for t in set(generated_ids[b].tolist()): 
            if t == MASK_TOKEN_ID: continue
            logits[b, :, t] /= penalty
    return logits

def inference(model, tokenizer, prompt_text, steps=None,debug_print=False, locking = True):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # # Tokenize the prompt
        # prompt_inputs_dummy = tokenizer(prompt_text, return_tensors="pt",
        #                                truncation=True,
        #                                max_length=TOTAL_SEQ_LEN-2)
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt",
                                  max_length=TOTAL_SEQ_LEN-2, truncation=True)
        prompt_ids = prompt_inputs.input_ids.to(device) 
        PROMPT_LEN = prompt_inputs.input_ids.size(1)
        B = prompt_ids.size(0)
        RESP_LEN = TOTAL_SEQ_LEN -PROMPT_LEN
        if not steps:
            steps = RESP_LEN
        
        # Initialize all response tokens as MASK
        response_ids = torch.full((B, RESP_LEN), MASK_TOKEN_ID, device=device)
        
        print(f"---- Inference: Diffusion Inference (steps: {steps}) ----")
        print("PROMPT", prompt_text)
        
        # Number of tokens to unlock per step
        tokens_per_step = max(1, RESP_LEN // steps)
        
        # Loop until all tokens are locked or we finish all steps
        for step in range(steps):
            # After updating response_ids with SEP padding
            combined_ids = torch.cat([prompt_ids, response_ids], dim=1)
            attention_mask = (combined_ids != tokenizer.pad_token_id).long()

            # Forward pass with updated mask
            logits = model(combined_ids, attention_mask=attention_mask)
            logits_resp = logits[:, PROMPT_LEN:, :]  # shape: [B, RESP_LEN, vocab_size]
            # TODO: not hardcode prompt_ids.size(1) as should be dynamic in the batch
            
            # Optional sampling penalty for various tokens
            if tokenizer.cls_token_id is not None:
                sep_penalty = -10.0  # you can tune this
                logits_resp[:, :, tokenizer.cls_token_id] += sep_penalty

            if steps < 5 and tokenizer.sep_token_id is not None:
                sep_penalty = -10.0  # you can tune this
                logits_resp[:, :, tokenizer.sep_token_id] += sep_penalty

            # Repition Penalty (Maybe have to check if it works corretly)
            logits_resp = apply_repetition_penalty(logits_resp, response_ids, penalty=10.0)

            # Convert logits to probabilities
            probs = torch.softmax(logits_resp, dim=-1)  # [B, RESP_LEN, vocab_size]
            top_probs, top_ids = probs.max(dim=-1)        # [B, RESP_LEN]
            
            # Get positions that are still MASK
            masked_positions = response_ids.eq(MASK_TOKEN_ID)  # [B, RESP_LEN]
            # For each example in the batch, lock in `tokens_per_step` tokens
            for b in range(B):
                if locking:
                    # Find indices of response tokens still masked
                    candidates = torch.nonzero(masked_positions[b]).squeeze(1)
                    if candidates.numel() == 0:
                        continue  # All tokens are locked
                    
                    # Limit the number of tokens to unlock in this step
                    if step < (steps -1):
                        num_to_unlock = min(tokens_per_step, candidates.numel())
                    else:
                        num_to_unlock = candidates.numel()
                else:
                    candidates = torch.nonzero(response_ids[b]).squeeze(1)
                    if step < (steps -1):
                        num_to_unlock =  min(tokens_per_step * (step + 1), candidates.numel())
                    else:
                        num_to_unlock = candidates.numel()
                # Select `num_to_unlock` candidates with highest confidence
                candidate_confidences = top_probs[b, candidates]
                best_indices = torch.topk(candidate_confidences, num_to_unlock).indices
                best_positions = candidates[best_indices]
                
                # Lock in the tokens at those positions
                response_ids[b, best_positions] = top_ids[b, best_positions]
            
            # Decode without skipping special tokens
            decoded = tokenizer.decode(response_ids[0], skip_special_tokens=False)
            # Remove other undesired special tokens (e.g., eos_token)
            if tokenizer.eos_token is not None:
                decoded = decoded.replace(tokenizer.eos_token, "")
            # Replace mask token with a gray-colored version
            decoded = decoded.replace(tokenizer.mask_token, "\033[90m" + tokenizer.mask_token + "\033[0m")
            if debug_print: print(f"Inference Step {step+1:02d}: {decoded}")
            
            # Stop early if no masked tokens remain
            if masked_positions.sum() == 0:
                break
    print("RESPONSE:", decoded)
    return decoded

if __name__ == "__main__":
    # Initialize the masked diffusion LM.
    embed_dim = 128
    num_layers = 4
    num_heads = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    model = MaskedDiffusionLM(VOCAB_SIZE, embed_dim, num_layers, num_heads).to(device)
    epochs_pretrain = 1000
    epochs_finetune = 1000
    pretrain = True
    finetune = True
    if pretrain:
        print("====== Pretraining ======")
        print(f"Train on {len(pretrain_dataset)} samples")
        # For pretraining, mask_prompt=True (random masking over entire sequence).
        train(model, tokenizer, pretrain_dataset, num_steps=len(pretrain_dataset) * epochs_pretrain, mask_prompt=True, accumulation_steps=1)
        torch.save(model.state_dict(), "model_pretrained.pth")
        print("Model saved as model_pretrained.pth")
    else:
        model.load_state_dict(torch.load("model_pretrained.pth", map_location=device))
        model.to(device)
    
    if finetune:
        print("====== Finetuning ======")
        print(f"Train on {len(finetune_dataset)} samples")
        # For pretraining, mask_prompt=True (random masking over entire sequence).
        train(model, tokenizer, finetune_dataset, num_steps=len(finetune_dataset) * epochs_finetune, mask_prompt=False, accumulation_steps=1)
        torch.save(model.state_dict(), "model_finetuned.pth")
        print("Model saved as model_finetuned.pth")
    elif not finetune and not pretrain:
        model.load_state_dict(torch.load("model_finetuned.pth", map_location=device))
        model.to(device)
    print("\n====== Inference ======")
    # During inference, we keep the prompt intact and update only the response.
    variations = [1,10,None]
    scores = []
    for variation in variations:
        scores.append(len(inference_dataset))
    for pair in inference_dataset:
        for index, i in enumerate(variations):
            decoded = inference(model, tokenizer, pair[0], steps=i)
            if decoded != pair[1]:
                print("Label:",pair[1],"\n\nVS\n\n",decoded)
                scores[index] -= 1
    for index, variation in enumerate(variations):
        score = scores[index]
        print(f"Score for {variation} diffusion steps: {score}/{len(inference_dataset)} ({round(float(score)*100/float(len(inference_dataset)))}%)")