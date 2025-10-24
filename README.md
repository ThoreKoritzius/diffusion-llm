# Diffusion-LLM

This project implements a simplified transformer-based LLM that generates SQL queries using a diffusion-like denoising process. It is an early experimental idea to blen diffusion models with transformer architectures, where condition the noise on language prompts.

![Diffusion Example](diffusion_example.gif)

## Features

- **Transformer-based architecture** for both encoding prompts and denoising.
- **Time-conditioned noise injection** for training and inference.
- **Cross-attention** to incorporate conditioning prompts.
- **Dual loss function** combining latent MSE and token-level cross-entropy.
- **Web Playground** Iterative denoising during inference to generate structured SQL outputs from noise.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
````

## Usages
Run the experiment by

```bash
python diffusion_llm.py
```

which trains the diffusion-llm architecture and performs inference on test examples.
