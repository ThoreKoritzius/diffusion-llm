import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fixed sequence length.
MAX_SEQ_LEN = 32

class DiffusionTransformerLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super().__init__()
        # Shared token embedding and fixed positional embedding.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, embed_dim))
        # Time conditioning: maps a normalized scalar [0,1] to an embedding.
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # Denoising transformer branch.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.denoise_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Cross-attention to inject conditioning.
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        # Instead of predicting the entire clean latent, we predict a residual.
        # This residual will be added to the fixed positional encoding.
        self.out_layer = nn.Linear(embed_dim, embed_dim)
        # A learned decoder: projects the predicted latent into token logits.
        self.vocab_decoder = nn.Linear(embed_dim, vocab_size)
        # A separate transformer to encode the prompt.
        cond_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.cond_transformer = nn.TransformerEncoder(cond_encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(embed_dim)

    def get_condition_encoding(self, cond_ids):
        """
        Encodes the conditioning prompt.
        Returns shape [seq_len, batch, embed_dim] for cross-attention.
        """
        cond_embed = self.token_embedding(cond_ids)  # [batch, seq_len, embed_dim]
        pos = self.pos_embedding[:, :cond_ids.size(1), :]
        cond_embed = cond_embed + pos
        cond_enc = self.cond_transformer(cond_embed.transpose(0, 1))  # [seq_len, batch, embed_dim]
        return cond_enc

    def forward(self, cond_ids, target_ids, time_cond):
        """
        Training forward pass.
        Inputs:
          cond_ids   : [batch, seq_len] tokens for the prompt.
          target_ids : [batch, seq_len] tokens for the target SQL query.
          time_cond  : a scalar in [0,1] representing the noise magnitude.
        Process:
          1. Compute the clean latent for the target from token & positional embeddings.
          2. Sample Gaussian noise scaled by time_cond, and add to the clean latent.
          3. Compute a time embedding from time_cond.
          4. Form the noisy latent as: latent_noisy + time_embedding.
          5. Pass through the denoising transformer and use cross-attention with condition.
          6. Through the output layer (and norm) predict the clean latent (x₀).
          7. Decode predicted clean latent to token logits.
        Returns:
          predicted_x0 : [batch, seq_len, embed_dim] – predicted clean latent.
          logits      : [batch, seq_len, vocab_size] – predicted token logits.
          latent_clean: [batch, seq_len, embed_dim] – reference clean latent.
        """
        batch = target_ids.size(0)
        # Compute clean latent: content + pos.
        target_embed = self.token_embedding(target_ids)
        latent_clean = target_embed + self.pos_embedding[:, :target_ids.size(1), :]
        # Add noise only to the content (implicitly, noise affects full latent).
        noise = torch.randn_like(latent_clean) * time_cond
        latent_noisy = latent_clean + noise
        # Time embedding.
        t_tensor = torch.tensor([[time_cond]], device=latent_noisy.device, dtype=latent_noisy.dtype)
        time_emb = self.time_mlp(t_tensor)  # [1, embed_dim]
        time_emb = time_emb.unsqueeze(1).expand(batch, latent_noisy.size(1), -1)
        latent_input = latent_noisy + time_emb
        # Process via Transformer.
        x = latent_input.transpose(0, 1)
        x = self.denoise_transformer(x)
        # Cross-attention with condition.
        cond_enc = self.get_condition_encoding(cond_ids)
        x, _ = self.cross_attn(query=x, key=cond_enc, value=cond_enc)
        x = x.transpose(0, 1)
        # Predict the residual – to be added to the fixed positional encoding.
        predicted_residual = self.out_layer(x)
        predicted_residual = self.norm(predicted_residual)
        # Reconstruct predicted clean latent by adding the fixed positional encoding.
        # (This enforces that the positions are preserved.)
        fixed_pos = self.pos_embedding[:, :x.size(1), :]
        predicted_x0 = predicted_residual + fixed_pos
        # Decode.
        logits = self.vocab_decoder(predicted_x0)
        return predicted_x0, logits, latent_clean

    def denoise_step(self, latent, cond_ids, time_cond):
        """
        Inference denoising step.
        Instead of a DDIM-style update, we predict x₀ and then update the latent
        via interpolation: latent <- (1 - gamma)*latent + gamma*predicted_x0.
        Inputs:
          latent    : current latent representation [B, L, D]
          cond_ids  : conditioning prompt tokens.
          time_cond : normalized time parameter in [0,1]
        Returns:
          predicted_x0 : predicted clean latent.
        """
        batch = latent.size(0)
        t_tensor = torch.tensor([[time_cond]], device=latent.device, dtype=latent.dtype)
        time_emb = self.time_mlp(t_tensor)
        time_emb = time_emb.unsqueeze(1).expand(batch, latent.size(1), -1)
        latent_input = latent + time_emb
        x = latent_input.transpose(0, 1)
        x = self.denoise_transformer(x)
        cond_enc = self.get_condition_encoding(cond_ids)
        x, _ = self.cross_attn(query=x, key=cond_enc, value=cond_enc)
        x = x.transpose(0, 1)
        predicted_residual = self.out_layer(x)
        predicted_residual = self.norm(predicted_residual)
        fixed_pos = self.pos_embedding[:, :x.size(1), :]
        predicted_x0 = predicted_residual + fixed_pos
        return predicted_x0

def train(model, tokenizer, dataset, num_steps=1000, lambda_ce=1.0, lambda_mse=1.0):
    """
    Trains on (prompt, target SQL) pairs with MSE loss (in latent space) and token cross-entropy loss
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(num_steps):
        cond_text, target_text = dataset[step % len(dataset)]
        cond_inputs   = tokenizer(cond_text, return_tensors="pt", padding="max_length",
                                    max_length=MAX_SEQ_LEN, truncation=True)
        target_inputs = tokenizer(target_text, return_tensors="pt", padding="max_length",
                                    max_length=MAX_SEQ_LEN, truncation=True)
        cond_ids = cond_inputs.input_ids  # [1, L]
        target_ids = target_inputs.input_ids  # [1, L]

        # Sample time_cond uniformly from [0, 1].
        time_cond = torch.empty(1).uniform_(0, 1).item()
        predicted_x0, logits, latent_clean = model(cond_ids, target_ids, time_cond)
        # Latent MSE loss.
        loss_mse = F.mse_loss(predicted_x0, latent_clean)
        # Token cross entropy loss.
        # Reshape logits to [B*L, vocab_size] and target_ids to [B*L].
        loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        loss = lambda_mse * loss_mse + lambda_ce * loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"[Training] Step {step} - Loss: {loss.item():.4f} (MSE: {loss_mse.item():.4f}, CE: {loss_ce.item():.4f})")

def inference(model, tokenizer, cond_text, num_steps=10, noise_scale=1.0, gamma=0.2):
    """
    Inference procedure:
        1. Start from pure noise.
        2. At each step, predict the clean latent (x₀), update latent via interpolation.
        3. Add back the fixed positional encoding to ensure correct word order.
        4. Decode tokens via the dedicated vocab_decoder.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        cond_inputs = tokenizer(cond_text, return_tensors="pt", padding="max_length",
                                    max_length=MAX_SEQ_LEN, truncation=True)
        cond_ids = cond_inputs.input_ids.to(device)  # [1, L]

        embed_dim = model.token_embedding.embedding_dim
        # Initialize latent as pure noise.
        latent = torch.randn(1, MAX_SEQ_LEN, embed_dim, device=device) * noise_scale
        pos_enc = model.pos_embedding[:, :MAX_SEQ_LEN, :]  # fixed positional encoding

        print("---- Inference: Iterative Denoising ----")
        for step in range(num_steps):
            # Vary the time condition (e.g., linearly from 1 to 0).
            t_normalized = 1 - (step / (num_steps - 1))
            predicted_x0 = model.denoise_step(latent, cond_ids, t_normalized)
            # Update latent via interpolation.
            latent = (1 - gamma) * latent + gamma * predicted_x0
            # FIX: Re-add the fixed positional encoding.
            latent_with_pos = latent + pos_enc

            # Decode tokens using the dedicated vocab_decoder.
            logits = model.vocab_decoder(latent_with_pos)  # [1, L, vocab_size]
            token_ids = torch.argmax(logits, dim=-1)  # [1, L]
            decoded = tokenizer.decode(token_ids[0], skip_special_tokens=True)
            print(f"Step {step+1:02d}: {decoded}")
    return decoded

if __name__ == "__main__":
    # Initialize tokenizer and model.
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    vocab_size = tokenizer.vocab_size
    embed_dim = 128
    num_layers = 2
    num_heads = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionTransformerLLM(vocab_size, embed_dim, num_layers, num_heads).to(device)

    dataset = [
        ("Count the rows of cars", "SELECT COUNT(*) FROM cars")
    ]
    print("====== Training ======")
    train(model, tokenizer, dataset, num_steps=500)
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")

    print("\n====== Inference ======")
    inference(model, tokenizer, "Count the rows of cars", num_steps=10, noise_scale=1.0, gamma=0.2)