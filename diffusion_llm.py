import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

# Fixed sequence length for both conditioning and target.
MAX_SEQ_LEN = 32

class DiffusionTransformerLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super().__init__()
        # Shared token embedding and fixed positional embedding.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, embed_dim))
        # Time conditioning: maps a scalar (normalized in [0,1]) to an embedding.
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # Denoising transformer branch.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.denoise_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Cross-attention branch to integrate conditioning.
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        # Final linear layer that now predicts the clean latent (x₀) instead of noise.
        self.out_layer = nn.Linear(embed_dim, embed_dim)
        # A learned decoder: projects the predicted latent into token logits.
        self.vocab_decoder = nn.Linear(embed_dim, vocab_size)
        # A separate transformer to encode the conditioning prompt.
        cond_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.cond_transformer = nn.TransformerEncoder(cond_encoder_layer, num_layers=1)
        # Optional normalization on the latent.
        self.norm = nn.LayerNorm(embed_dim)

    def get_condition_encoding(self, cond_ids):
        """
        Encodes the conditioning prompt.
        Returns shape [seq_len, batch, embed_dim] for cross-attention.
        """
        cond_embed = self.token_embedding(cond_ids)  # [batch, seq_len, embed_dim]
        pos = self.pos_embedding[:, :cond_ids.size(1), :]
        cond_embed = cond_embed + pos
        cond_enc = self.cond_transformer(cond_embed.transpose(0,1))  # [seq_len, batch, embed_dim]
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
        # (1) Compute clean latent.
        target_embed = self.token_embedding(target_ids)
        latent_clean = target_embed + self.pos_embedding[:, :target_ids.size(1), :]  # [B, L, D]
        # (2) Add noise.
        noise = torch.randn_like(latent_clean) * time_cond
        latent_noisy = latent_clean + noise
        # (3) Time embedding.
        t_tensor = torch.tensor([[time_cond]], device=latent_noisy.device, dtype=latent_noisy.dtype)
        time_emb = self.time_mlp(t_tensor)  # [1, D]
        time_emb = time_emb.unsqueeze(1).expand(batch, latent_noisy.size(1), -1)
        # (4) Combine noisy latent with time embedding.
        latent_input = latent_noisy + time_emb
        # (5) Process via denoising transformer.
        x = latent_input.transpose(0, 1)   # [L, B, D]
        x = self.denoise_transformer(x)
        # Inject the conditioning (cross-attention).
        cond_enc = self.get_condition_encoding(cond_ids)  # [L, B, D]
        x, _ = self.cross_attn(query=x, key=cond_enc, value=cond_enc)
        x = x.transpose(0,1)  # [B, L, D]
        # (6) Predict the clean latent.
        predicted_x0 = self.out_layer(x)
        predicted_x0 = self.norm(predicted_x0)
        # (7) Decode to token logits.
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
        predicted_x0 = self.out_layer(x)
        predicted_x0 = self.norm(predicted_x0)
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
      1. Given a conditioning prompt, initialize the latent as pure noise.
      2. For each denoising step, predict the clean latent (x₀) and update the current latent
         via an interpolation:
             new_latent = (1 - gamma) * latent + gamma * predicted_x0
      3. Decode the current latent via the dedicated vocab_decoder.
    The idea is that over several steps the latent will gradually move from noise toward a latent
    that decodes to the desired SQL query.
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
        print("---- Inference: Iterative Denoising ----")
        for step in range(num_steps):
            # Linearly vary the time condition from 1 down to 0.
            t_normalized = 1 - (step / (num_steps - 1))
            predicted_x0 = model.denoise_step(latent, cond_ids, t_normalized)
            # Update latent using interpolation.
            latent = (1 - gamma) * latent + gamma * predicted_x0
            # Decode tokens.
            logits = model.vocab_decoder(latent)  # [1, L, vocab_size]
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
    train(model, tokenizer, dataset, num_steps=800)
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")

    print("\n====== Inference ======")
    inference(model, tokenizer, "Count the rows of cars", num_steps=10, noise_scale=1.0, gamma=0.2)