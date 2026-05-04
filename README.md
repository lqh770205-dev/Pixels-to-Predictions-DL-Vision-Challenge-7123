# Deep-Learning-CS-GY-6953-ECE-GY-7123
# Pixels to Predictions — DL Vision Challenge (Course Project)

Multimodal multiple-choice QA with **`HuggingFaceTB/SmolVLM-500M-Instruct`**, efficient fine-tuning (**LoRA / DoRA**, trainable parameters **≤ 5M** per competition rules), and **answer scoring via token log-likelihood** (letter + optional choice-text ensemble). Primary artifact: **`final_project.ipynb`**.

---

## Repository layout

| Path | Role |
|------|------|
| `final_project.ipynb` | **Main pipeline**: data discovery → LoRA/DoRA training (optional) → inference → `submission.csv` |
| `starter_notebook.ipynb` | Starter / exploration (prompt demo, optional LoRA notes) |
| `train.csv`, `val.csv`, `test.csv` | Labels / IDs when running **locally** (mirror competition CSVs) |
| `sample_submission.csv` | Required submission column template |
| `images/` | Image folders referenced by `image_path` in CSVs (when data is unpacked locally) |

On **Kaggle**, competition files usually appear under  
`/kaggle/input/competitions/<dataset-name>/`  
(or similar); the notebook **auto-detects** `train.csv` or use **`DATA_DIR_OVERRIDE`**.

---

## Environment & dependencies

- **Python 3.10+** recommended.
- **PyTorch** with CUDA if using GPU.
- Install before running the main notebook:

```bash
pip install "peft>=0.13.0" transformers accelerate pillow
```

On **Kaggle**, add a cell at the top if needed:

```python
# !pip install -q "peft>=0.13.0"
```

---

## Full pipeline (documentation / grading checklist)

The following end-to-end stages are implemented **in code** and summarized here for reproducibility:

1. **Resolve data root** — `discover_data_dir()` finds `train.csv` (Kaggle input, local cwd, or `DATA_DIR_OVERRIDE`).
2. **Load splits** — `train_df`, `val_df`, `test_df`; parse JSON `choices`.
3. **Optional train merge** — If `USE_VAL_FOR_TRAINING = True`, training uses **train ∪ val** rows.
4. **Base model** — `AutoModelForImageTextToText.from_pretrained(MODEL_ID)` in FP16 on GPU.
5. **Adapter search** — LoRA/DoRA on LLM linear layers (`PREFERRED_LORA_SUFFIXES`), rank `r` chosen so **trainable parameters ≤ `MAX_TRAINABLE_PARAMS`** (default 5_000_000).
6. **Training** (when `RUN_TRAINING = True`) — Causal LM loss with labels masked to **answer tokens only**; AdamW + cosine schedule with warmup; gradient accumulation; optional gradient checkpointing.
7. **Checkpoint write** — `model.save_pretrained(ADAPTER_DIR)`, `processor.save_pretrained(ADAPTER_DIR)` (PEFT adapter files: `adapter_config.json`, `adapter_model.safetensors`, etc.).
8. **Optional caption cache** — If `USE_IMAGE_CAPTION = True`, captions saved to `CAPTION_CACHE_PATH` (e.g. `/kaggle/working/caption_cache.json`).
9. **Inference** — For each test row: image + prompt → score each choice (letter LL + weighted choice-text LL) → argmax → integer answer index.
10. **Merge (optional)** — `merge_and_unload()` after training when supported, for faster inference.
11. **Output** — `submission.csv` with columns **`id`**, **`answer`** (0-based choice index).

---

## Checkpoints & artifacts (where things are saved)

| Artifact | Default location (Kaggle) | Contents |
|----------|---------------------------|----------|
| **PEFT adapter** | `/kaggle/working/smolvlm-mcqa-lora/` | LoRA/DoRA weights + adapter config; reload with `RUN_TRAINING = False` |
| **Caption cache** | `/kaggle/working/caption_cache.json` | Per-`image_path` caption strings (only if captions enabled) |
| **Submission** | `/kaggle/working/submission.csv` | Competition submission |

Configurable in the notebook **`CONFIG`** section: `ADAPTER_DIR`, `CAPTION_CACHE_PATH`.

---

## Reproducibility (training & inference)

### Random seed

All stochastic components use a single seed **`SEED = 42`** (declared in `final_project.ipynb`):

- `torch.manual_seed(SEED)`
- `numpy.random.seed(SEED)`
- `random.seed(SEED)`
- `torch.cuda.manual_seed_all(SEED)` when CUDA is available  

Inference scoring uses **`do_sample=False`** where generation is used (captions). Full bitwise reproducibility across machines/CUDA builds is **not** guaranteed by PyTorch/Hugging Face, but runs are **seed-controlled** for repeatability.

### Training setup (how to reproduce a training run)

1. Place competition data so `train.csv` / `images/` resolve (or set **`DATA_DIR_OVERRIDE`**).
2. Set **`RUN_TRAINING = True`**.
3. Ensure **`ADAPTER_DIR`** is writable (Kaggle: `/kaggle/working/...`).
4. Run **all cells** in `final_project.ipynb` top to bottom (**Save & Run All** on Kaggle).
5. After completion, verify **`adapter_config.json`** exists under **`ADAPTER_DIR`**.

### Inference-only setup (reuse checkpoints without retraining)

1. Train once as above **or** copy a saved adapter folder into **`ADAPTER_DIR`**.
2. Set **`RUN_TRAINING = False`**.
3. Confirm **`ADAPTER_DIR / adapter_config.json`** exists.
4. Run the notebook — loads base model + adapter, writes **`submission.csv`**.

---

## Recommended Kaggle run (GPU T4 ×2)

1. **Add data** — Competition dataset attached to the notebook.
2. **Settings** → **Accelerator** → **GPU T4 ×2** → **Save** → **Session → Restart Session**.
3. Optional pip cell for **`peft`** (see above).
4. Adjust **`CONFIG`** if needed (defaults tuned for ~16 GB VRAM per GPU; see comments for OOM fallback).
5. **Save & Run All** — Submit **`submission.csv`** from **Output**.

---

## Competition constraints (summary)

- Model checkpoint: **`HuggingFaceTB/SmolVLM-500M-Instruct`** (official HF).
- **Trainable parameters ≤ 5 million** (enforced by rank search + target modules).
- **No external datasets** beyond the competition release.
- Submission: **`submission.csv`** with **`id`**, **`answer`** (integer index).

---

## Troubleshooting

| Issue | Suggested action |
|-------|------------------|
| `train.csv` not found | Set **`DATA_DIR_OVERRIDE`** to the folder containing **`train.csv`** |
| CUDA OOM | Reduce **`IMG_SIZE`** (e.g. 320), set **`TRY_DORA = False`**, **`USE_IMAGE_CAPTION = False`** |
| `ImportError: peft` | Run **`pip install "peft>=0.13.0"`** |

---

## Course submission: adapter weights on GitHub

For reproducibility and grading, bundle the **trained PEFT adapter** (everything produced under `ADAPTER_DIR`, including `adapter_config.json`, `adapter_model.safetensors`, tokenizer/processor files, and optionally `best_val/` after training) into **`adapter.zip`**.

**Suggested placement:** attach **`adapter.zip`** to a [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) on this repository, or commit the adapter folder (keep repo size reasonable). After uploading, add the Release URL or folder path to your report’s **Reproducibility** section so points match this README.

Inference-only rerun: unzip so `ADAPTER_DIR` points at the folder containing `adapter_config.json`, then set **`RUN_TRAINING = False`** and run the notebook.

---

## Author notes

Detailed inline documentation appears in the **first Markdown cell** of `final_project.ipynb`. This README is the **canonical reproducibility** reference for graders and peers.

