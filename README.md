# Pixels to Predictions — DL Vision Challenge 

Multimodal multiple-choice QA with **`HuggingFaceTB/SmolVLM-500M-Instruct`**, efficient fine-tuning (**LoRA / DoRA**, trainable parameters **≤ 5M** per competition rules), and **answer scoring via token log-likelihood** (letter + optional choice-text ensemble). Primary artifact: **`final_project.ipynb`**.

**→ To reproduce training from scratch, jump to [Retrain from scratch (step-by-step)](#retrain-from-scratch-step-by-step).**  
**→ ACL-format report (PDF source):** `report/report_main.tex` (figures: run `python report/generate_figures.py`).

---

## What to submit (minimal)

For course hand-in, the usual bundle is only:

| Bundle item | Purpose |
|-------------|---------|
| **`adapter.zip`** | Trained PEFT adapter (`adapter_config.json`, `adapter_model.safetensors`, tokenizer/processor files). See [Course submission](#course-submission-adapter-weights-on-github). |
| **`final_project.ipynb`** | Main code: training + inference → `submission.csv`. |
| **`README.md`** | How to install deps, point `ADAPTER_DIR`, and rerun inference (this file). |

That is enough for someone to load your weights and reproduce **`submission.csv`** without your full repo clone.

**Nice to include (small, helps graders):** `requirements.txt` — keeps `pip install -r requirements.txt` aligned with what you used (PyTorch still installed separately; see **Environment & dependencies** below).

---

## Repository layout (full clone — optional extras)

If you keep the whole GitHub project (development + report), the tree looks like this. Nothing below is required in the **minimal** bundle unless your instructor asks for the written report source.

```
Deep-Learning-CS-GY-6953-ECE-GY-7123/
├── final_project.ipynb      # Single entry point: train + infer → submission.csv
├── README.md
├── requirements.txt
├── starter_notebook.ipynb   # Optional: exploration / starter prompts
├── report/                  # Optional: LaTeX report + figures for the PDF write-up
│   ├── report_main.tex
│   ├── generate_figures.py
│   └── figures/
├── train.csv, val.csv, test.csv   # Local/Kaggle copies of competition CSVs (not shipped in adapter.zip)
├── sample_submission.csv
└── images/                  # Local: unzip competition images here when not on Kaggle
```

On **Kaggle**, competition files usually appear under `/kaggle/input/competitions/<dataset-name>/`; the notebook **auto-detects** `train.csv` or use **`DATA_DIR_OVERRIDE`**.

---

## Environment & dependencies

- **Python 3.10+** recommended.
- **Install PyTorch first** for your platform (CUDA vs CPU): https://pytorch.org/get-started/locally/
- Then install project dependencies from this repo:

```bash
cd /path/to/this/repo
pip install -r requirements.txt
```

Equivalent one-liner (if you prefer not to use `requirements.txt`):

```bash
pip install "peft>=0.13.0" transformers accelerate pillow pandas numpy tqdm
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

### Retrain from scratch (step-by-step)

Use this checklist to satisfy **reproducible training** grading criteria (environment + data + exact procedure):

| Step | Action |
|------|--------|
| 1 | Clone this repository and `cd` into it. |
| 2 | Create a Python **3.10+** venv/conda env. |
| 3 | Install **PyTorch** matching your GPU driver from pytorch.org, then `pip install -r requirements.txt`. |
| 4 | Download competition **CSV + images** from Kaggle; place `train.csv`, `val.csv`, `test.csv` and `images/` so paths match `image_path`, **or** set **`DATA_DIR_OVERRIDE`** to that folder’s absolute path in `final_project.ipynb`. |
| 5 | In **`CONFIG`**: `RUN_TRAINING = True`, `SEED = 42`, leave defaults unless you are running an ablation. |
| 6 | Run **every cell** from top to bottom (same order as authored — do not skip). |
| 7 | Confirm artifacts: **`ADAPTER_DIR/adapter_config.json`**, **`adapter_model.safetensors`**, tokenizer files; optional **`ADAPTER_DIR/best_val/`** if validation improved. |
| 8 | Optional: zip the adapter folder as **`adapter.zip`** for Release upload (see [Course submission](#course-submission-adapter-weights-on-github)). |

**Expected training behavior:** console prints `Training rows: …`, per-epoch loss, `val ep… accuracy`, and `Saved … adapter`. Hyperparameters are **only** changed via `CONFIG` at the top of the notebook (single source of truth).

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

## Documentation map (report vs repo)

| Topic | Where |
|-------|--------|
| Method, splits, ablations, error analysis | `report/report_main.tex` → compiled PDF |
| Commands to install, train, infer | **This README** (sections above) |
| Line-by-line hyperparameters | `CONFIG` block in `final_project.ipynb` + Appendix table in PDF |
| Adapter files for graders | `adapter.zip` on GitHub Release or committed adapter folder |

## Author notes

Detailed inline documentation appears in the **first Markdown cell** of `final_project.ipynb`. This README is the **canonical reproducibility** reference for graders and peers.
