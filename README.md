# Diffusion-LLM

A masked language model implementing **diffusion-style text denoising** for SQL generation, leveraging the `RoBERTa architectur`. This project includes the training pipeline using a HF text-to-SQL dataset and ships with a Gradio-based inference studio to visualize the iterative diffusion process.

The core idea: Provide a `--prompt` (natural language question) and a `--context` (relevant table schema). The LLM then iteratively refines and decodes the corresponding SQL query over multiple diffusion steps which can be adjusted at inference. Therefore we are shift from sequential auto-regression to decode multiple tokens at once in one step.

![Diffusion Example](examples/diffusion_example.gif)

---

## Features

- **Diffusion-Style Masking:** Trains and infers by repeatedly masking variable SQL spans and predicting tokens in an iterative denoising schedule.
- **Customizable Training:** Easily modify masking schedules, batch size, and data columns to experiment with various text denoising ablations.
- **Live Playground:** Intuitive web GUI to run and visualize the generation process, with GIF download for animation of each diffusion step.

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Diffusion-Style SQL Model

Modify hyperparameters inside `src/train.py` as needed, then run:

```bash
python src/train.py
```

This will:
- Download a synthetic text-to-SQL dataset
- Train a masked language model on diffusion-style denoising over SQL spans
- Save the trained model to disk, inside `diffusion-sql` folder
- Log data to `wandb`

### 3. Run the Gradio Inference Studio

```bash
python src/gradio_inference.py
```

- Stream each denoising step with live SQL viewport
- Scrub through intermediate snapshots
- Auto-fill schema context from a CSV upload, or paste your own `CREATE TABLE` DDL
- Adjust steps/top-k/top-p/max length/model directory directly from the UI
