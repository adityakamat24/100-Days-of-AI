# 100 Days of AI – Colab T4 Edition
_Free tools • Single 6 GB GPU • 60–90 min a day_

---

## How to use this guide
* **One notebook per day.** Copy today’s section, paste into a fresh Colab, run top‑to‑bottom.
* **Keep a public repo.** Commit the notebook and a 1‑line `README.md` log (`Day 7 ✅ – Monte‑Carlo π`).
* **Fridays are heavier.** They’re mini‑projects; block ~2 h or split Sat/Sun.

Every day has:
* **Why it matters** – the “so what”.
* **Prep** – installs / downloads.
* **Tasks** – numbered, runnable on a free Tesla T4.
* **Proof** – what to save or tweet.
* **Stretch** – optional level‑up.

---

## Phase 1 — Groundwork (Days 1‑14)

### Day 1 – Hello, GPU
**Why** Know your hardware.  
**Prep**
```bash
!nvidia-smi
```
**Tasks** 1) Run, note “Tesla T4”. 2) `import torch; print(torch.cuda.is_available())`  
**Proof** Screenshot in repo.  
**Stretch** Log VRAM every 2 s for 1 min.

### Day 2 – NumPy two‑layer perceptron
**Why** Feel back‑prop before frameworks.  
**Prep** `import numpy as np, sklearn.datasets`  
**Tasks** 1) Load Iris. 2) Build 4‑16‑3 MLP with NumPy. 3) SGD 1 000 steps, ≥ 85 % acc.  
**Proof** Notebook `day02_numpy_mlp.ipynb`.  
**Stretch** Replace finite‑diff with analytic grads.

### Day 3 – Pandas Titanic clean‑up
**Why** Tabular ETL is daily bread.  
**Prep** `import pandas as pd`  
**Tasks** 1) Load `train.csv` from Kaggle Titanic. 2) Fill nulls, encode sex. 3) Save `clean.csv`.  
**Proof** Diff showing new file.  
**Stretch** EDA heat‑map of correlations.

### Day 4 – Matplotlib heat‑maps
**Why** Quick visuals.  
**Prep** `import matplotlib.pyplot as plt, seaborn as sns`  
**Tasks** 1) Correlation matrix of Titanic. 2) Save `heatmap.png`.  
**Proof** PNG in repo.  
**Stretch** Add age histogram subplot.

### Day 5 – Linear‑algebra in code
**Why** Matrix skills power DL.  
**Prep** None  
**Tasks** Implement `matmul`, `sigmoid`, unit test vs NumPy.  
**Proof** `day05_linear.py` passes tests.  
**Stretch** Write a tiny benchmarks table.

### Day 6 – Numerical gradient checker
**Why** Trust but verify.  
**Prep** Reuse Day 2 MLP.  
**Tasks** 1) Write `grad_check(f, params)`. 2) Compare numeric vs analytic.  
**Proof** Log max diff < 1e‑4.  
**Stretch** Vectorise finite‑diff pass.

### Day 7 – Monte‑Carlo π
**Why** Probability + simulation.  
**Prep** `import random, math`  
**Tasks** Estimate π with 1 M darts, plot error curve vs samples.  
**Proof** Figure in repo.  
**Stretch** Use GPU via CuPy.

### Day 8 – Metrics from scratch
**Why** Never rely blindly on libraries.  
**Prep** `sklearn.model_selection.train_test_split`  
**Tasks** Implement accuracy, precision, recall; unit‑test on toy preds.  
**Proof** `metrics.py` + tests.  
**Stretch** Add F1 and ROC‑AUC.

### Day 9 – Logistic regression baseline
**Why** Benchmark before deep nets.  
**Prep** `from sklearn.linear_model import LogisticRegression`  
**Tasks** 1) Fit on Titanic clean.csv. 2) Report accuracy.  
**Proof** Markdown result.  
**Stretch** Plot ROC curve.

### Day 10 – Grid search hyperparams
**Why** Tuning basics.  
**Prep** `from sklearn.model_selection import GridSearchCV`  
**Tasks** Grid C ∈ {0.1,1,10}, penalty l1/l2.  
**Proof** Best params + score.  
**Stretch** RandomisedSearch 30 configs.

### Day 11 – Git branching
**Why** Team hygiene.  
**Prep** `git` CLI.  
**Tasks** 1) Create `feature/day11`. 2) PR into main with reviewer.  
**Proof** GitHub PR URL.  
**Stretch** Resolve merge conflict.

### Day 12 – VS Code debug
**Why** Step through models.  
**Prep** Open Day 2 MLP in VS Code.  
**Tasks** Set breakpoint on loss, inspect grads.  
**Proof** Screenshot.  
**Stretch** Watch VRAM panel live.

### Day 13 – Foundations quiz
**Why** Reinforce theory.  
**Prep** `pytest`  
**Tasks** Write 5 auto‑graded Qs NumPy/Stats.  
**Proof** `pytest -q` passes.  
**Stretch** Add CI badge.

### Day 14 – Mini‑project: Titanic Kaggle ≥ 0.80
**Why** First complete loop.  
**Prep** Kaggle API.  
**Tasks** Feature‑engineer, model, submit.  
**Proof** Leaderboard screenshot.  
**Stretch** Explain top 3 features via SHAP.

---

## Phase 2 — Classic ML (Days 15‑28)

### Day 15 – k‑NN from scratch
**Why** Lazy learners intuition.  
**Prep** Iris dataset.  
**Tasks** Code k‑NN (k=5), achieve ≥ 95 %.  
**Proof** Notebook.  
**Stretch** KD‑Tree speed‑up.

### Day 16 – Decision trees visualised
**Why** Interpretable models.  
**Prep** `sklearn.tree`, `graphviz`.  
**Tasks** Depth‑3 tree on Iris, export PNG.  
**Proof** PNG in repo.  
**Stretch** Plot feature importance.

### Day 17 – Random forest OOB
**Why** Bagging power.  
**Prep** see guide earlier.  
**Tasks** Adult RF 100 trees, log OOB vs test.  
**Proof** Markdown table.  
**Stretch** Tune n_estimators curve.

### Day 18 – Gradient boosting intro
**Why** Strong tabular baseline.  
**Prep** `xgboost`  
**Tasks** XGBoost on Titanic, tune rounds.  
**Proof** Accuracy logged.  
**Stretch** Early stopping.

### Day 19 – SVM margins
**Why** Kernel tricks.  
**Prep** `sklearn.svm.SVC`  
**Tasks** 2‑D moons dataset, plot support vectors.  
**Proof** Figure.  
**Stretch** Try RBF vs linear.

### Day 20 – PCA MNIST scatter
**Why** Dimensionality reduction.  
**Prep** `sklearn.decomposition.PCA`  
**Tasks** 784→50→2, plot.  
**Proof** PNG.  
**Stretch** t‑SNE comparison.

### Day 21 – k‑means clustering
**Why** Unsupervised basics.  
**Prep** Mall customers CSV.  
**Tasks** Elbow plot, choose k, label plot.  
**Proof** Notebook.  
**Stretch** Silhouette score.

### Day 22 – Feature engineering pipeline
**Why** Clean preprocessing.  
**Prep** `ColumnTransformer`  
**Tasks** Scale numeric, one‑hot categorical.  
**Proof** `preprocessor.pkl`.  
**Stretch** Add log‑transform.

### Day 23 – Scikit‑learn pipeline
**Why** One‑liner model + preprocessing.  
**Prep** Reuse Day 22.  
**Tasks** Pipeline + RF, cross‑val 5‑fold.  
**Proof** CV scores.  
**Stretch** Joblib dump.

### Day 24 – SHAP interpretability
**Why** Explain decisions.  
**Prep** `shap`  
**Tasks** TreeExplainer on 1 k rows; plot top 10 features.  
**Proof** PNG.  
**Stretch** Dependence plot.

### Day 25 – Bias & fairness
**Why** Responsible ML.  
**Prep** Adult data gender column.  
**Tasks** Compute demographic parity diff.  
**Proof** Metric number.  
**Stretch** Mitigate via reweighing.

### Day 26 – Model persistence
**Why** Reuse without retrain.  
**Prep** `joblib`  
**Tasks** Save Day 23 pipeline, load, predict.  
**Proof** Checksum of file.  
**Stretch** Version stamp in metadata.

### Day 27 – Flask prediction API
**Why** Serving 101.  
**Prep** `flask`, saved model.  
**Tasks** Route `/predict`, runs JSON → output.  
**Proof** `curl` result.  
**Stretch** Dockerfile.

### Day 28 – Mini‑project: Dockerised service
**Why** Deploy end‑to‑end.  
**Tasks** 1) Docker build. 2) Run container, hit endpoint.  
**Proof** Docker Hub image link.  
**Stretch** GH Actions build.

---

## Phase 3 — Deep Learning Basics (Days 29‑42)

### Day 29 – PyTorch tensors & autograd
**Why** Core DL primitive.  
**Tasks** Re‑implement Day 2 MLP in PyTorch.  
**Proof** Notebook.  
**Stretch** Compare param counts.

### Day 30 – Manual training loop
**Why** Understand every op.  
**Tasks** 3‑layer MLP on Fashion‑MNIST, 5 epochs.  
**Proof** Loss curve.  
**Stretch** Add val split.

### Day 31 – GPU speed‑up
*(see detailed earlier)*

### Day 32 – Optimisers showdown
**Why** SGD vs Adam.  
**Tasks** Train same net with both, plot losses.  
**Proof** Plot.  
**Stretch** Try RMSProp.

### Day 33 – Regularisation tricks
**Why** Prevent overfit.  
**Tasks** Add dropout 0.3 + weight decay 1e‑4.  
**Proof** Val accuracy table.  
**Stretch** BatchNorm.

### Day 34 – Tiny CNN on CIFAR‑10
**Why** Images intro.  
**Tasks** 3‑conv CNN, 60 %+.  
**Proof** Accuracy.  
**Stretch** Confusion matrix.

### Day 35 – Data augmentation gains
**Why** Cheap accuracy.  
**Tasks** Random crop & flip, boost to 65 %.  
**Proof** Curve.  
**Stretch** Cutout.

### Day 36 – TensorBoard logging
**Why** Visual debugging.  
**Tasks** Log losses, images every epoch.  
**Proof** Screenshot.  
**Stretch** Projection embeddings.

### Day 37 – Transfer learning MobileNet‑V2
**Why** Small & strong.  
**Tasks** Fine‑tune on Oxford Flowers (10‑class subset).  
**Proof** Acc ≥ 80 %.  
**Stretch** Freeze encoder.

### Day 38 – Mixed precision AMP
**Why** Speed + memory.  
**Tasks** Enable `autocast` & `GradScaler`, compare step time.  
**Proof** Timing table.  
**Stretch** BFloat16 cpu.

### Day 39 – Checkpoints & resume
**Why** Long runs resilience.  
**Tasks** Save best val acc; resume mid‑epoch.  
**Proof** Console logs.  
**Stretch** Multiple checkpoints keep‑last 3.

### Day 40 – PyTorch Lightning intro
**Why** Boilerplate killer.  
**Tasks** Wrap Day 37 as LightningModule.  
**Proof** Train runs.  
**Stretch** `trainer.predict`.

### Day 41 – Callbacks early stop
**Why** Good defaults.  
**Tasks** EarlyStopping(patience=3) + ModelCheckpoint.  
**Proof** Callback logs.  
**Stretch** LearningRateMonitor.

### Day 42 – Mini‑project: Dogs‑vs‑Cats32
**Why** Cap images block.  
**Tasks** 85 %+ accuracy on 32×32 subset.  
**Proof** Leaderboard.  
**Stretch** Grad‑CAM demo gif.

---

## Phase 4 — Vision Extras (Days 43‑50)

### Day 43 – Grad‑CAM heat‑maps
**Prep** `pytorch-grad-cam`  
**Tasks** Overlay on Dog/Cat image.  
**Proof** PNG.  
**Stretch** Batch mode.

### Day 44 – YOLOv8‑nano fine‑tune
**Tasks** 5 epochs on 10 custom images.  
**Proof** mAP printed.  
**Stretch** Export ONNX.

### Day 45 – Tiny U‑Net segmentation
*(see detailed earlier)*

### Day 46 – Label Studio basics
**Tasks** Label 5 images, export COCO JSON.  
**Proof** JSON in repo.

### Day 47 – ONNX export
**Tasks** Export MobileNet‑V2 to ONNX, run `onnxruntime`.  
**Proof** Latency ms.  
**Stretch** Opt level.

### Day 48 – Int8 quantisation
**Tasks** Dynamic quant MobileNet, compare Top‑1 drop.  
**Proof** Acc table.

### Day 49 – Channel pruning
**Tasks** Prune 30 %, fine‑tune 1 epoch.  
**Proof** Model size diff.

### Day 50 – Mini‑project: Streamlit Grad‑CAM app
**Tasks** Upload image → heat‑map.  
**Proof** Deployed Streamlit share URL.

---

## Phase 5 — NLP & Transformers (Days 51‑70)

We keep every day runnable on a Colab T4 by using small checkpoints (Distil* models, TinyBERT, TinyLlama 4‑bit) and tiny datasets (IMDb 5 k reviews, WikiText‑2 10 k lines). Sequence length ≤ 128, batch size ≤ 4.

### Day 51 – Text cleaning & tokenisation
**Why it matters**  
Rubbish in → rubbish out. Clean text once, reuse forever.

**Prep**
```python
import pandas as pd, re, nltk; nltk.download('stopwords')
```

**Step‑by‑step**
1. Create `clean_text()` that lowercases, strips punctuation, and removes English stop‑words.
2. Load IMDb 5 k reviews (`datasets.load_dataset('imdb', split='train[:5000]')`).
3. Apply cleaner, save to `imdb_clean.csv`.

**Proof of work**  
Commit `imdb_clean.csv` and `clean_text.py`.

**Level‑up (optional)**  
Plot top‑20 most frequent tokens after cleaning.

---
### Day 52 – Word2Vec embeddings
**Why it matters**  
Distributed representations beat one‑hot vectors.

**Prep**
```python
!pip install -q gensim
dataset already loaded in Day 51 cell
```

**Step‑by‑step**
1. Use `gensim.models.Word2Vec(sentences, vector_size=50, window=5, min_count=5, workers=2)`.
2. Train for 5 epochs.
3. Query `model.wv.most_similar('movie', topn=5)`.

**Proof of work**  
Screenshot of similar words + save `word2vec.model`.

**Level‑up (optional)**  
Visualise embeddings with TSNE 2‑D scatter of 200 words.

---
### Day 53 – GRU sentiment classifier
**Why it matters**  
Recurrent nets still useful and tiny enough for T4.

**Prep**
```python
!pip install -q torchtext
torch, torchtext
```

**Step‑by‑step**
1. Use `torchtext` to build vocab from Day 51 CSV (token length ≤128).
2. Define GRU (embedding 100d → GRU 128 → FC 1).
3. Train 3 epochs, batch 4, BCE loss.
4. Evaluate accuracy on 1 k IMDb test lines.

**Proof of work**  
Notebook `day53_gru_sentiment.ipynb` with accuracy ≥ 0.75.

**Level‑up (optional)**  
Add dropout 0.3 and compare.

---
### Day 54 – Scaled‑dot attention in NumPy
**Why it matters**  
Understand core of Transformers by coding the math yourself.

**Prep**
```python
import numpy as np
```

**Step‑by‑step**
1. Implement `attention(Q,K,V)` per Vaswani.
2. Verify shapes: Q,K,V = random (2,4,64).
3. Compare manual result vs `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.1).

**Proof of work**  
Python script passes shape & value diff < 1e‑5.

**Level‑up (optional)**  
Add causal mask and verify upper‑triangle zeros.

---
### Day 55 – Tiny Transformer encoder
**Why it matters**  
See how attention + feed‑forward combine.

**Prep**
```python
import torch, torch.nn as nn
```

**Step‑by‑step**
1. Build `EncoderBlock` with multi‑head attention (dim 128, heads 4) and 2‑layer MLP.
2. Stack 2 blocks; forward dummy batch (16×20 tokens).
3. Parameter count should be < 1 M.

**Proof of work**  
Print param total and forward output shape.

**Level‑up (optional)**  
Train on synthetic copy‑task until loss < 0.02.

---
### Day 56 – Positional encodings visualised
**Why it matters**  
Transformers need position info.

**Prep**
```python
import matplotlib.pyplot as plt, torch, math
```

**Step‑by‑step**
1. Implement sinusoidal encodings for positions 0‑99, dim 64.
2. Plot 4 selected dimensions against position.

**Proof of work**  
PNG plot committed.

**Level‑up (optional)**  
Add learnable positional embeddings and compare magnitude histograms.

---
### Day 57 – Pre‑train mini‑BERT (4 M params)
**Why it matters**  
Hands‑on masked language modelling without huge compute.

**Prep**
```python
!pip install -q transformers datasets
```

**Step‑by‑step**
1. Tokenizer = `bert-base-uncased` (limit vocab to 20 k tokens via `tokenizer.train_new_from_iterator`).
2. Build 4‑layer BERT (`transformers.BertConfig` hidden 256, heads 4).
3. Load 10 k lines WikiText‑2, create 15 % masked inputs.
4. Train 2 epochs, batch 8, fp16.

**Proof of work**  
Training loss curve in notebook, final loss < 3.0.

**Level‑up (optional)**  
Run inference to fill masks in sample sentence.

---
### Day 58 – Fine‑tune BERT on SST‑2
**Why it matters**  
Show transfer learning boosts small data.

**Prep**
```python
!pip install -q datasets transformers
```

**Step‑by‑step**
1. Load SST‑2 (GLUE) train[:5 k].
2. Replace classification head (2 classes).
3. Fine‑tune 3 epochs, lr 2e‑5.

**Proof of work**  
Validation accuracy ≥ 0.80 logged.

**Level‑up (optional)**  
Try differential learning rates (encoder 1e‑5, head 1e‑4).

---
### Day 59 – Train BPE tokenizer
**Why it matters**  
Tight tokenizer improves generation quality.

**Prep**
```python
!pip install -q tokenizers datasets
```

**Step‑by‑step**
1. Collect IMDb + Wiki lines (100 k lines).
2. Train `ByteLevelBPETokenizer` vocab 16 k.
3. Save `vocab.json`, `merges.txt`.

**Proof of work**  
Files committed with total size shown.

**Level‑up (optional)**  
Benchmark encoding speed vs Hugging Face tokenizer.

---
### Day 60 – DistilGPT‑2 jokes fine‑tune
**Why it matters**  
First generative model; fun & fast.

**Prep**
```python
!(see prep commands in doc above)
```

**Step‑by‑step**
1. Load `distilgpt2`.
2. Dataset: Reddit Jokes 5 k entries.
3. Trainer 2 epochs, batch 2, fp16.
4. Generate 3 jokes for prompt.

**Proof of work**  
Notebook + screenshot of best joke.

**Level‑up (optional)**  
Implement LoRA via `peft` and compare perplexity.

---
### Day 61 – Decoding strategies
**Why it matters**  
Sampling controls creativity.

**Prep**
```python
(imports shown in steps)
```

**Step‑by‑step**
1. Implement greedy, top‑k (k=50), nucleus p=0.9 functions using Transformers generate.
2. Evaluate diversity: unique 4‑gram ratio across 50 generations for prompt 'Once upon a time'.
3. Chart diversity vs coherence (manual quick rating).

**Proof of work**  
Markdown with diversity numbers & opinion.

**Level‑up (optional)**  
Add beam search (beam=4) and compare.

---
### Day 62 – LoRA haiku fine‑tune
**Why it matters**  
Parameter‑efficient FT where VRAM is scarce.

**Prep**
```python
!pip install -q peft transformers datasets
```

**Step‑by‑step**
1. Apply LoRA (rank 8) to DistilGPT‑2.
2. Dataset: 3 k English haikus.
3. Train 3 epochs; only LoRA params ~1 M.

**Proof of work**  
Size of LoRA adapter file (<20 MB) and sample haiku.

**Level‑up (optional)**  
Merge LoRA into base weights and re‑export.

---
### Day 63 – PEFT vs full fine‑tune eval
**Why it matters**  
Quantify savings.

**Prep**
```python
(imports shown in steps)
```

**Step‑by‑step**
1. Compare LoRA model (Day 62) to baseline full fine‑tune (Day 60) on perplexity over 500 unseen Reddit jokes.
2. Measure VRAM during training via `torch.cuda.max_memory_allocated()`.

**Proof of work**  
Table: perplexity & VRAM.

**Level‑up (optional)**  
Plot train wall‑clock time.

---
### Day 64 – Build FAISS index
**Why it matters**  
RAG basics.

**Prep**
```python
!pip install -q faiss-cpu
```

**Step‑by‑step**
1. Take 20 k Wikipedia paragraph embeddings via MiniLM.
2. Build FAISS flat index.
3. Single query latency printed.

**Proof of work**  
Index size on disk and latency ms.

**Level‑up (optional)**  
Swap to IVF index and compare.

---
### Day 65 – LangChain Q&A bot
**Why it matters**  
Glue components quickly.

**Prep**
```python
!pip install -q langchain openai
```

**Step‑by‑step**
1. LangChain RetrievalQA with FAISS index + DistilGPT‑2 as generator.
2. Ask 3 factual questions.

**Proof of work**  
Answers screenshot.

**Level‑up (optional)**  
Streamlit UI.

---
### Day 66 – Prompt engineering few‑shot
**Why it matters**  
Prompts > params sometimes.

**Prep**
```python
(imports shown in steps)
```

**Step‑by‑step**
1. Create zero‑shot, one‑shot, and three‑shot prompts for sentiment classification with OpenAI `gpt-3.5-turbo` (if key) or DistilGPT‑2.
2. Evaluate accuracy on 100 SST‑2 samples.

**Proof of work**  
Accuracy table.

**Level‑up (optional)**  
Use automatic prompt generation via `prompt-tools`.

---
### Day 67 – Chain‑of‑thought reasoning
**Why it matters**  
Show better reasoning with CoT.

**Prep**
```python
(imports shown in steps)
```

**Step‑by‑step**
1. Use HF `tiiuae/falcon-7b-instruct` in 4‑bit via AutoGPTQ if available; else call OpenAI.
2. Solve 10 grade‑school math problems with and without `Let's think step by step`.
3. Record correct counts.

**Proof of work**  
Markdown results.

**Level‑up (optional)**  
Try self‑consistency decoding.

---
### Day 68 – Add classification head
**Why it matters**  
Repurpose MLM.

**Prep**
```python
(imports shown in steps)
```

**Step‑by‑step**
1. Take mini‑BERT from Day 57.
2. Add new linear head for emotion 6‑class dataset (dair-ai/emotion).
3. Fine‑tune 3 epochs.

**Proof of work**  
Validation macro‑F1.

**Level‑up (optional)**  
Freeze encoder and train only head, compare.

---
### Day 69 – Toxicity filtering
**Why it matters**  
Safety matters.

**Prep**
```python
!pip install -q detoxify
```

**Step‑by‑step**
1. Generate 100 jokes with Day 60 model.
2. Score toxicity; filter >0.5.
3. Compute % removed.

**Proof of work**  
Markdown stats.

**Level‑up (optional)**  
Fine‑tune on filtered vs raw jokes, compare.

---
### Day 70 – Mini‑project: Telegram Shakespeare bot
**Why it matters**  
Put your text model in the wild.

**Prep**
```python
!pip install -q python-telegram-bot
```

**Step‑by‑step**
1. Bot uses LangChain RAG (Day 65) + Shakespeare style prompt.
2. Host via `ngrok` or polling.
3. Exchange at least one user conversation.

**Proof of work**  
Screencap of chat and GitHub repo link.

**Level‑up (optional)**  
Deploy to Hugging Face Spaces.

---
## Phase 6 — LLM Internals & Optimisation (Days 71‑85)

These days dive under the hood. Models: DistilGPT‑2 (82 M), TinyLlama‑1.1 B 4‑bit (~5 GB), and synthetic toy nets. Heavy math days run on CPU with small tensors.
### Day 71 – Self‑attention math & KV shapes
**Why it matters**  
Derive and prove instead of rote‑using.

**Prep**
```python
import torch, math
```

**Step‑by‑step**
1. Write equations for Q,K,V sizes (batch b, seq s, dim d). Derive output shape.
2. Create random tensors (b=2,s=5,d=16) and compute attention manually.
3. Assert your result equals PyTorch op.

**Proof of work**  
Markdown derivation + assertion pass.

**Level‑up (optional)**  
Show that caching K,V halves compute for incremental decoding.

---
### Day 72 – KV‑cache timing
**Why it matters**  
Empirically measure benefit.

**Prep**
```python
(see imports earlier)
```

**Step‑by‑step**
1. Load DistilGPT‑2.
2. Generate 50 tokens with and without `use_cache=False`.
3. Time each with `time.perf_counter()`.

**Proof of work**  
Table tokens/s.

**Level‑up (optional)**  
Plot length vs speed graph (10,20,50,100).

---
### Day 73 – Speculative decoding
**Why it matters**  
Speed up sampling using cheap draft model.

**Prep**
```python
(imports as in Day 60/71)
```

**Step‑by‑step**
1. Draft model: 15 M param student from Day 78 (placeholder tiny‑gpt2).
2. Implement algorithm per Chen 2023.
3. Measure wall‑time speed‑up over 200 tokens.

**Proof of work**  
Notebook with speed‑up ≥ 1.4×.

**Level‑up (optional)**  
Vary acceptance threshold k and plot.

---
### Day 74 – Gradient checkpointing demo
**Why it matters**  
Swap memory for compute.

**Prep**
```python
import torch, torch.utils.checkpoint
```

**Step‑by‑step**
1. Use Day 34 tiny CNN.
2. Train 1 epoch with and without `torch.utils.checkpoint` wrappers on conv blocks.
3. Capture VRAM via `torch.cuda.max_memory_allocated()`.

**Proof of work**  
Markdown with memory savings %.

**Level‑up (optional)**  
Compare epoch time.

---
### Day 75 – torch.compile benchmark
**Why it matters**  
New PyTorch compiler often free speed.

**Prep**
```python
!pip install -q torch==2.1
```

**Step‑by‑step**
1. Load TinyLlama‑1.1 B in 4‑bit QLoRA (`bitsandbytes`).
2. Benchmark 10 forward passes vanilla vs `torch.compile(model)`.

**Proof of work**  
Throughput table.

**Level‑up (optional)**  
Try different compile back‑ends (inductor vs nvFuser).

---
### Day 76 – ZeRO offload on CPU
**Why it matters**  
Sharding tricks for multi‑GPU replaced by offload on single GPU.

**Prep**
```python
!pip install -q deepspeed
```

**Step‑by‑step**
1. Configure DeepSpeed ZeRO‑3 with CPU offload for mini‑BERT.
2. Train 100 steps MLM.
3. Log GPU memory in megabytes.

**Proof of work**  
Notebook log showing <3 GB usage.

**Level‑up (optional)**  
Compare training speed vs normal FP16.

---
### Day 77 – QLoRA TinyLlama
**Why it matters**  
Most popular low‑resource fine‑tune method.

**Prep**
```python
!pip install -q peft bitsandbytes transformers accelerate
```

**Step‑by‑step**
1. Load TinyLlama‑1.1 B with 4‑bit quant.
2. Apply LoRA rank‑8 on two linear layers.
3. Fine‑tune 500 steps on Alpaca 100.

**Proof of work**  
Memory usage <6 GB & training loss curve.

**Level‑up (optional)**  
Export merged adapter and run inference.

---
### Day 78 – Knowledge distillation student
**Why it matters**  
Shrink for edge devices.

**Prep**
```python
(imports previous)
```

**Step‑by‑step**
1. Teacher: DistilGPT‑2.
2. Student: 15 M param 2‑layer GPT.
3. Train for 1 epoch on WikiText 2 subset using KL loss.

**Proof of work**  
Plot student vs teacher perplexity.

**Level‑up (optional)**  
Quantise student to int8 and benchmark.

---
### Day 79 – Toy Mixture‑of‑Experts
**Why it matters**  
Routing concepts for efficiency.

**Prep**
```python
import torch.nn as nn
```

**Step‑by‑step**
1. Build gating network and 2 linear experts (25 k params each).
2. Route random input batch; verify only 1 expert active per token.

**Proof of work**  
Print routing histogram.

**Level‑up (optional)**  
Add load‑balancing loss.

---
### Day 80 – Sparse attention mask
**Why it matters**  
Handle long sequences without OOM.

**Prep**
```python
import torch
```

**Step‑by‑step**
1. Implement block‑sparse mask (stride 32) for 64‑token batch.
2. Multiply with random attention scores and verify zeros.

**Proof of work**  
Matrix visualisation as image.

**Level‑up (optional)**  
Benchmark vs dense on same size.

---
### Day 81 – RLHF bandit summariser
**Why it matters**  
Intro to preference learning.

**Prep**
```python
!pip install -q trl datasets
```

**Step‑by‑step**
1. Use `trl` PPO trainer with DistilGPT‑2.
2. Task: summarise Reddit post (100 samples) reward = ROUGE‑1 improvement vs lead‑3 baseline.
3. Train 500 steps.

**Proof of work**  
Reward curve rising.

**Level‑up (optional)**  
Swap to KL‑penalised decoding and compare.

---
### Day 82 – KL penalty tuning
**Why it matters**  
Stabilise policy updates.

**Prep**
```python
(continue from Day 81)
```

**Step‑by‑step**
1. Vary KL coefficient {0.01,0.1,1}.
2. Plot average reward vs coefficient.

**Proof of work**  
Chart committed.

**Level‑up (optional)**  
Find sweet spot coefficient.

---
### Day 83 – Evaluation harness
**Why it matters**  
Automate model comparison.

**Prep**
```python
!pip install -q evaluate nltk rouge_score
```

**Step‑by‑step**
1. Write `evaluate.py` CLI that computes BLEU, ROUGE‑L on given model & dataset.
2. Run on student, teacher from Day 78.

**Proof of work**  
CLI output screenshot.

**Level‑up (optional)**  
Add Winograd schema challenge (WSC) accuracy.

---
### Day 84 – Model card documentation
**Why it matters**  
Transparency requirements.

**Prep**
```python

```

**Step‑by‑step**
1. Draft `README.md` with model details: training data, intended use, limitations, licenses.
2. Include eval metrics from Day 83.

**Proof of work**  
Markdown file committed.

**Level‑up (optional)**  
Generate JSON‑schema model card.

---
### Day 85 – Mini‑project: Publish to Hugging Face Hub
**Why it matters**  
Share and version your work.

**Prep**
```python
!pip install -q huggingface_hub
```

**Step‑by‑step**
1. `huggingface-cli login`.
2. Push student model + card.
3. Create demo Space with gradio chatbot.

**Proof of work**  
Provide model repo URL and Space link.

**Level‑up (optional)**  
Add inference widget to README.

---
## Phase 7 — Lightweight MLOps & Serving (Days 86‑94)

No Kubernetes or big cloud bills—just the essentials you can demo from Colab or a cheap VPS.
### Day 86 – Data versioning with DVC
**Why it matters**  
Reproducibility starts at data.

**Prep**
```python
!pip install -q dvc[s3]
```

**Step‑by‑step**
1. Init DVC repo.
2. Track dataset `imdb_clean.csv`.
3. Commit & push to GitHub.
4. Remote = free DVC cloud (`dvc remote add -d myremote s3://...` optional).

**Proof of work**  
Git commit diff showing `.dvc` files.

**Level‑up (optional)**  
Enable pre‑commit hook to auto‑add artifacts.

---
### Day 87 – CI test & slim Docker build
**Why it matters**  
Automation catches breakage.

**Prep**
```python

```

**Step‑by‑step**
1. Write `pytest` for one utility.
2. Add `.github/workflows/ci.yml` running tests + `docker build --target slim`.
3. Badge in README.

**Proof of work**  
Green CI checkmark screenshot.

**Level‑up (optional)**  
Cache docker layers for faster build.

---
### Day 88 – Serve model with TorchServe
**Why it matters**  
Production‑grade REST out of the box.

**Prep**
```python
!pip install -q torchserve torch-model-archiver
```

**Step‑by‑step**
1. Archive DistilGPT‑2 joke model.
2. Launch TorchServe on Colab (`--ncs` single process).
3. POST request with curl.

**Proof of work**  
Latency number and response example.

**Level‑up (optional)**  
Set concurrency 4 and stress‑test with `ab`.

---
### Day 89 – FastAPI JWT gateway
**Why it matters**  
Add auth + streaming.

**Prep**
```python
!pip install -q fastapi uvicorn python-multipart pyjwt
```

**Step‑by‑step**
1. FastAPI endpoint `/generate` validating JWT.
2. Inside call TorchServe and stream tokens via SSE.
3. Run locally on Colab public URL via `ngrok`.

**Proof of work**  
Screencast of curl with token returning stream.

**Level‑up (optional)**  
Add rate‑limit middleware.

---
### Day 90 – Resource logging with psutil
**Why it matters**  
Monitor before you need pagers.

**Prep**
```python
import psutil, time, json
```

**Step‑by‑step**
1. Write script sampling CPU, RAM, GPU util every 5 s for 5 min during FastAPI call loop.
2. Save to `metrics.csv`.

**Proof of work**  
CSV committed + matplotlib line plot.

**Level‑up (optional)**  
Add Prometheus pushgateway integration.

---
### Day 91 – Traffic splitter canary
**Why it matters**  
Safe releases even on localhost.

**Prep**
```python
bash script
```

**Step‑by‑step**
1. Run old model on port 8080, new on 8081.
2. Bash `weight=10` send 10 % requests to 8081 for 5 min.
3. Compare response latency.

**Proof of work**  
Histogram plot latencies.

**Level‑up (optional)**  
Wrap script into Makefile target.

---
### Day 92 – Data drift detection
**Why it matters**  
Models rot silently.

**Prep**
```python
!pip install -q evidently
```

**Step‑by‑step**
1. Load live vs training IMDb reviews subset.
2. `evidently` KS test on sentiment distribution.
3. Output HTML report.

**Proof of work**  
HTML saved in repo.

**Level‑up (optional)**  
Set threshold alarm message.

---
### Day 93 – Offline A/B analytics
**Why it matters**  
Decide winner without production infra.

**Prep**
```python
import pandas as pd
```

**Step‑by‑step**
1. Simulate 10 k impressions JSON log.
2. Compute CTR for variant A vs B.
3. Chi‑square test p‑value.

**Proof of work**  
Markdown conclusion.

**Level‑up (optional)**  
Plot daily CTR over 7 days.

---
### Day 94 – Cost per 1 K tokens notebook
**Why it matters**  
Know your $$ before shipping.

**Prep**
```python
import pandas as pd
```

**Step‑by‑step**
1. Create table of provider prices (OpenAI, Replicate, own GPU at $0.35/hr).
2. Calculate $/1 K tokens for Jokebot (Day 60) given throughput.
3. Plot bar chart.

**Proof of work**  
Notebook and bar chart image.

**Level‑up (optional)**  
Add sensitivity analysis for GPU spot pricing.

---
## Phase 8 — Capstone Sprint (Days 95‑100)

A week to polish and ship a portfolio‑ready AI product that mixes at least two skills you learned.
### Day 95 – Ideation & spec
**Why it matters**  
Clear scope prevents rabbit holes.

**Prep**
```python

```

**Step‑by‑step**
1. Brainstorm 3 ideas using Day 66 prompt engineering.
2. Pick one with MoSCOW priority matrix.
3. Write one‑pager (problem, users, data, success metric).

**Proof of work**  
Commit `capstone_idea.md`.

**Level‑up (optional)**  
Collect feedback from 2 peers and revise.

---
### Day 96 – Repo skeleton & Kanban
**Why it matters**  
Plan before code.

**Prep**
```python

```

**Step‑by‑step**
1. Init GitHub repo `capstone-<name>`.
2. Create standard folders: `data`, `src`, `notebooks`, `app`.
3. Add GitHub Projects board with backlog → todo → done.

**Proof of work**  
Repo URL + screenshot board.

**Level‑up (optional)**  
Add Semantic‑release config.

---
### Day 97 – MVP demo
**Why it matters**  
Ship something that works end‑to‑end.

**Prep**
```python

```

**Step‑by‑step**
1. Implement minimal pipeline (e.g., upload text → summary).
2. Record 2‑min Loom demo.
3. Open public issue for feedback.

**Proof of work**  
Loom link & open issue.

**Level‑up (optional)**  
Gather latency numbers and attach to README.

---
### Day 98 – Polish & docs
**Why it matters**  
Professionalism shows.

**Prep**
```python

```

**Step‑by‑step**
1. Add unit tests (≥ 60 % coverage).
2. Write Dockerfile.
3. Create `docs/` with mkdocs or Sphinx.

**Proof of work**  
CI passing badge and Doc site URL.

**Level‑up (optional)**  
Automate pre‑commit linting.

---
### Day 99 – Public launch
**Why it matters**  
Real users give real learning.

**Prep**
```python

```

**Step‑by‑step**
1. Deploy to Hugging Face Spaces or Render.
2. Write Twitter/X thread + LinkedIn post.
3. Add screenshots/gifs to repo.

**Proof of work**  
Links to launch posts + deployed URL.

**Level‑up (optional)**  
Submit to relevant subreddit for feedback.

---
### Day 100 – Retrospective & next steps
**Why it matters**  
Solidify learning and plan future growth.

**Prep**
```python

```

**Step‑by‑step**
1. Write blog post (Medium/dev.to) covering journey, wins, pain points.
2. List top 5 focus areas next 6 months.
3. Thank contributors in README.

**Proof of work**  
Blog post link committed.

**Level‑up (optional)**  
Turn blog into conference talk proposal.

---
---

## Survival Checklist

* **Mount Google Drive** (`from google.colab import drive; drive.mount('/content/gdrive')`) for anything you can’t lose.
* **Reduce batch & seq length** whenever you see `CUDA out of memory`.
* Call `torch.cuda.empty_cache()` between heavy cells.
* If GPU queue is busy: `Runtime ▸ Change runtime type ▸ None` (CPU) and let long jobs run while you sleep.
* Keep two tabs: one for coding, one for `nvidia-smi -l 2` watching VRAM in real time.
* Blog & tweet progress—accountability is rocket fuel.
