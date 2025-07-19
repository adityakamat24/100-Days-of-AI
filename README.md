
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

## Phase 5 — NLP & Transformers (Days 51‑70)

*(Days abbreviated for brevity, each keeps same format)*

51 Text cleaning & tokenisation | Proof: cleaned CSV  
52 Train Word2Vec IMDb 50d | Proof: nearest words list  
53 GRU sentiment vs logistic | Proof: accuracy table  
54 Scaled‑dot attention NumPy demo | Proof: shape asserts pass  
55 Tiny Transformer encoder 2‑layer | Proof: overfits toy copy‑task  
56 Positional encoding visual plot | Proof: sinusoid figure  
57 Pre‑train 4 M‑param mini‑BERT 2 epochs | Proof: MLM loss  
58 Fine‑tune BERT SST‑2 | Proof: F1 score  
59 Train BPE tokenizer | Proof: vocab.json size  
60 DistilGPT2 jokes fine‑tune (detailed earlier)  
61 Decoding: greedy vs top‑k vs nucleus | Proof: diversity metrics  
62 LoRA haiku model | Proof: loss curve  
63 PEFT eval vs full fine‑tune | Proof: table  
64 FAISS 20 k para RAG | Proof: retrieval latency  
65 LangChain Q&A over index | Proof: example answer  
66 Prompt engineering few‑shot | Proof: perplexity numbers  
67 Chain‑of‑thought math | Proof: accuracy bump  
68 Add classification head mini‑BERT | Proof: downstream task score  
69 Toxicity filtering Detoxify | Proof: before/after chart  
70 Mini‑project: Telegram Shakespeare bot | Proof: chat screenshot

---

## Phase 6 — LLM Internals (Days 71‑85)

71 Self‑attention math & KV shapes | Proof: derivation markdown  
72 KV cache timing DistilGPT2 | Proof: ms/token table  
73 Speculative decoding (detailed)  
74 Grad checkpointing tiny CNN | Proof: RAM vs time chart  
75 Torch.compile speed test TinyLlama 4‑bit | Proof: throughput  
76 ZeRO offload CPU demo | Proof: param shards printed  
77 QLoRA TinyLlama 1.1 B | Proof: VRAM usage  
78 Knowledge distillation 15 M student | Proof: perplexity drop  
79 Toy MoE 2 experts | Proof: routing plot  
80 Sparse attention mask demo | Proof: attention matrix image  
81 RL summariser bandit | Proof: ROUGE improvement  
82 KL penalty tuning | Proof: reward curve  
83 Evaluation harness BLEU/ROUGE | Proof: CLI output  
84 Model card draft | Proof: markdown file  
85 Mini‑project: Publish to HF Hub | Proof: model URL

---

## Phase 7 — Lightweight MLOps (Days 86‑94)

86 DVC/Git‑LFS dataset & ckpts | Proof: `.dvc` files  
87 GH Actions unit test + slim Docker build | Proof: CI badge  
88 TorchServe joke model (detailed)  
89 FastAPI JWT streaming | Proof: local curl chunked  
90 psutil log CPU/GPU util | Proof: CSV log  
91 Traffic splitter shell ‑‑ canary 10 % | Proof: count mismatch  
92 Data drift KS test CSVs | Proof: p‑value  
93 Offline A/B CTR in pandas | Proof: bar chart  
94 Cost per 1 K tokens calc | Proof: notebook

---

## Phase 8 — Capstone (Days 95‑100)

95 Idea one‑pager | Proof: PDF  
96 Build skeleton repo & issues | Proof: GitHub link  
97 MVP demo Loom 2 min | Proof: Loom URL  
98 Polish tests + docs + Docker | Proof: CI green  
99 Public launch tweet/thread | Proof: tweet URL  
100 Retrospective blog post | Proof: medium/dev.to link

---

### Survival checklist
* **Mount Google Drive** for checkpoints.  
* **Batch = 1** and gradient accumulation if OOM.  
* **`torch.cuda.empty_cache()`** between big loops.  
* **CPU fallback overnight** when necessary.

Happy hacking. See you at Day 100!
