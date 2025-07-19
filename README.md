# 100 Days of AI

A structured 100-day journey to master AI and machine learning using free Google Colab T4 GPUs. Designed for intermediate learners, this guide focuses on practical, hands-on projects that fit into 60–90 minutes daily, leveraging a single 6 GB GPU.

---

## Overview

This repository contains a 100-day curriculum to build AI and ML skills, from foundational concepts to advanced deep learning and lightweight MLOps. Each day includes a focused task, clear objectives, and proof of work to track progress. The guide is optimized for Google Colab's free Tesla T4 GPU, ensuring accessibility without expensive hardware.

### Key Features
- **Daily Commitment**: 60–90 minutes per day, with Fridays reserved for ~2-hour mini-projects.
- **Free Tools**: Uses Google Colab's T4 GPU (6 GB VRAM) and open-source libraries.
- **Structured Phases**: Organized into 8 phases covering groundwork, classic ML, deep learning, vision, NLP, LLM internals, MLOps, and a capstone project.
- **Public Accountability**: Encourages maintaining a public GitHub repo with daily commits and a concise `README.md` log (e.g., `Day 7 – Monte-Carlo π`).
- **Scalable Challenges**: Each day includes a main task and an optional "Level-up" for advanced exploration.

### Target Audience
- Intermediate learners with basic Python and ML knowledge.
- Developers aiming to build a portfolio of practical AI projects.
- Self-learners seeking a structured path to master AI on a budget.

---

## How to Use This Guide

1. **One Notebook Per Day**: Copy the day's code and instructions into a fresh Google Colab notebook. Run cells top-to-bottom.
2. **Maintain a Public Repo**: Commit each day's notebook and update the `README.md` with a one-line log (e.g., `Day 7 – Monte-Carlo π`).
3. **Fridays Are Mini-Projects**: Allocate ~2 hours or split across Saturday/Sunday for deeper tasks.
4. **Track Progress**: Save proofs of work (e.g., screenshots, plots, model files) in your repo.
5. **Share Your Journey**: Blog or post progress on platforms like X or LinkedIn for accountability.

### Daily Structure
Each day includes:
- **Why It Matters**: Explains the task's real-world relevance.
- **Prep**: Lists required libraries or setup commands.
- **Tasks**: Step-by-step instructions, runnable on a T4 GPU.
- **Proof of Work**: Specific deliverables to commit (e.g., notebook, plot, or metric).
- **Level-up (Optional)**: Advanced challenge for deeper learning.

---

## Curriculum

### Phase 1: Groundwork (Days 1–14)
Build foundational skills in hardware, linear algebra, data processing, and version control.

- **Day 1 – Hello, GPU**  
  **Why**: Understand your hardware.  
  **Prep**: `!nvidia-smi`  
  **Tasks**: Confirm Tesla T4, check `torch.cuda.is_available()`.  
  **Proof**: Screenshot of `nvidia-smi` output.  
  **Level-up**: Log VRAM every 2 seconds for 1 minute.

- **Day 2 – NumPy Two-Layer Perceptron**  
  **Why**: Grasp backpropagation fundamentals.  
  **Prep**: `import numpy as np, sklearn.datasets`  
  **Tasks**: Build a 4-16-3 MLP on Iris dataset, train with SGD to ≥85% accuracy.  
  **Proof**: Commit `day02_numpy_mlp.ipynb`.  
  **Level-up**: Use analytic gradients instead of finite differences.

- **Day 3 – Pandas Titanic Cleanup**  
  **Why**: Master tabular data preprocessing.  
  **Prep**: `import pandas as pd`  
  **Tasks**: Load Kaggle Titanic `train.csv`, fill nulls, encode categorical variables, save `clean.csv`.  
  **Proof**: Git diff showing `clean.csv`.  
  **Level-up**: Create a correlation heatmap.

- **Day 4 – Matplotlib Heatmaps**  
  **Why**: Visualize data insights.  
  **Prep**: `import matplotlib.pyplot as plt, seaborn as sns`  
  **Tasks**: Plot Titanic correlation matrix, save `heatmap.png`.  
  **Proof**: Commit PNG.  
  **Level-up**: Add age histogram subplot.

- **Day 5 – Linear Algebra in Code**  
  **Why**: Matrix operations underpin deep learning.  
  **Prep**: None  
  **Tasks**: Implement `matmul`, `sigmoid`, and unit tests vs NumPy.  
  **Proof**: Commit `day05_linear.py` with passing tests.  
  **Level-up**: Create a benchmarks table.

- **Day 6 – Numerical Gradient Checker**  
  **Why**: Verify gradient computations.  
  **Prep**: Reuse Day 2 MLP.  
  **Tasks**: Write `grad_check(f, params)`, compare numeric vs analytic gradients.  
  **Proof**: Log max difference < 1e-4.  
  **Level-up**: Vectorize finite-difference pass.

- **Day 7 – Monte-Carlo π**  
  **Why**: Learn probability and simulation.  
  **Prep**: `import random, math`  
  **Tasks**: Estimate π with 1M darts, plot error vs sample size.  
  **Proof**: Commit error curve plot.  
  **Level-up**: Use CuPy for GPU acceleration.

- **Day 8 – Metrics from Scratch**  
  **Why**: Understand evaluation metrics deeply.  
  **Prep**: `sklearn.model_selection.train_test_split`  
  **Tasks**: Implement accuracy, precision, recall; unit-test on toy predictions.  
  **Proof**: Commit `metrics.py` with tests.  
  **Level-up**: Add F1 and ROC-AUC.

- **Day 9 – Logistic Regression Baseline**  
  **Why**: Establish a simple ML benchmark.  
  **Prep**: `from sklearn.linear_model import LogisticRegression`  
  **Tasks**: Fit model on Titanic `clean.csv`, report accuracy.  
  **Proof**: Markdown with accuracy.  
  **Level-up**: Plot ROC curve.

- **Day 10 – Grid Search Hyperparameters**  
  **Why**: Learn hyperparameter tuning.  
  **Prep**: `from sklearn.model_selection import GridSearchCV`  
  **Tasks**: Grid search C ∈ {0.1, 1, 10}, penalty l1/l2.  
  **Proof**: Commit best parameters and score.  
  **Level-up**: Try RandomizedSearchCV with 30 configs.

- **Day 11 – Git Branching**  
  **Why**: Master team collaboration workflows.  
  **Prep**: `git` CLI  
  **Tasks**: Create `feature/day11` branch, submit PR to main.  
  **Proof**: GitHub PR URL.  
  **Level-up**: Resolve a merge conflict.

- **Day 12 – VS Code Debugging**  
  **Why**: Debug models effectively.  
  **Prep**: Open Day 2 MLP in VS Code.  
  **Tasks**: Set breakpoint on loss, inspect gradients.  
  **Proof**: Screenshot of debugger.  
  **Level-up**: Monitor VRAM live in VS Code.

- **Day 13 – Foundations Quiz**  
  **Why**: Reinforce theoretical knowledge.  
  **Prep**: `pytest`  
  **Tasks**: Write 5 auto-graded NumPy/stats questions.  
  **Proof**: `pytest -q` passing output.  
  **Level-up**: Add CI badge to repo.

- **Day 14 – Mini-Project: Titanic Kaggle ≥0.80**  
  **Why**: Complete an end-to-end ML pipeline.  
  **Prep**: Kaggle API  
  **Tasks**: Feature engineer, train model, submit to leaderboard.  
  **Proof**: Leaderboard screenshot.  
  **Level-up**: Explain top 3 features with SHAP.

---

### Phase 2: Classic ML (Days 15–28)
Deepen understanding of traditional machine learning algorithms and techniques.

- **Day 15 – k-NN from Scratch**  
  **Why**: Understand lazy learning.  
  **Prep**: Iris dataset  
  **Tasks**: Code k-NN (k=5), achieve ≥95% accuracy.  
  **Proof**: Commit notebook.  
  **Level-up**: Implement KD-Tree for speedup.

- **Day 16 – Decision Trees Visualized**  
  **Why**: Learn interpretable models.  
  **Prep**: `sklearn.tree`, `graphviz`  
  **Tasks**: Train depth-3 tree on Iris, export PNG.  
  **Proof**: Commit PNG.  
  **Level-up**: Plot feature importance.

- **Day 17 – Random Forest OOB**  
  **Why**: Explore bagging benefits.  
  **Prep**: Adult dataset  
  **Tasks**: Train 100-tree RF, compare OOB vs test accuracy.  
  **Proof**: Markdown table.  
  **Level-up**: Tune `n_estimators` curve.

- **Day 18 – Gradient Boosting Intro**  
  **Why**: Master strong tabular models.  
  **Prep**: `xgboost`  
  **Tasks**: Train XGBoost on Titanic, tune rounds.  
  **Proof**: Log accuracy.  
  **Level-up**: Implement early stopping.

- **Day 19 – SVM Margins**  
  **Why**: Understand kernel methods.  
  **Prep**: `sklearn.svm.SVC`  
  **Tasks**: Train on 2D moons dataset, plot support vectors.  
  **Proof**: Commit plot.  
  **Level-up**: Compare RBF vs linear kernel.

- **Day 20 – PCA MNIST Scatter**  
  **Why**: Learn dimensionality reduction.  
  **Prep**: `sklearn.decomposition.PCA`  
  **Tasks**: Reduce MNIST 784→50→2, plot scatter.  
  **Proof**: Commit PNG.  
  **Level-up**: Compare with t-SNE.

- **Day 21 – k-Means Clustering**  
  **Why**: Master unsupervised learning.  
  **Prep**: Mall customers dataset  
  **Tasks**: Create elbow plot, choose k, plot labels.  
  **Proof**: Commit notebook.  
  **Level-up**: Compute silhouette score.

- **Day 22 – Feature Engineering Pipeline**  
  **Why**: Streamline preprocessing.  
  **Prep**: `ColumnTransformer`  
  **Tasks**: Scale numeric, one-hot encode categorical features.  
  **Proof**: Commit `preprocessor.pkl`.  
  **Level-up**: Add log-transform.

- **Day 23 – Scikit-learn Pipeline**  
  **Why**: Combine preprocessing and modeling.  
  **Prep**: Reuse Day 22  
  **Tasks**: Build pipeline with RF, run 5-fold cross-validation.  
  **Proof**: Commit CV scores.  
  **Level-up**: Save pipeline with `joblib`.

- **Day 24 – SHAP Interpretability**  
  **Why**: Explain model decisions.  
  **Prep**: `shap`  
  **Tasks**: Use TreeExplainer on 1k rows, plot top 10 features.  
  **Proof**: Commit PNG.  
  **Level-up**: Create dependence plot.

- **Day 25 – Bias & Fairness**  
  **Why**: Ensure responsible ML.  
  **Prep**: Adult dataset (gender column)  
  **Tasks**: Compute demographic parity difference.  
  **Proof**: Commit metric value.  
  **Level-up**: Mitigate bias via reweighing.

- **Day 26 – Model Persistence**  
  **Why**: Reuse models efficiently.  
  **Prep**: `joblib`  
  **Tasks**: Save/load Day 23 pipeline, predict.  
  **Proof**: Commit file checksum.  
  **Level-up**: Add version metadata.

- **Day 27 – Flask Prediction API**  
  **Why**: Learn model serving basics.  
  **Prep**: `flask`, saved model  
  **Tasks**: Create `/predict` endpoint, test with `curl`.  
  **Proof**: Commit `curl` result.  
  **Level-up**: Add Dockerfile.

- **Day 28 – Mini-Project: Dockerized Service**  
  **Why**: Deploy end-to-end.  
  **Tasks**: Build/run Docker container, hit endpoint.  
  **Proof**: Docker Hub image link.  
  **Level-up**: Add GitHub Actions build.

---

### Phase 3: Deep Learning Basics (Days 29–42)
Transition to deep learning with PyTorch, focusing on GPU-optimized workflows.

- **Day 29 – PyTorch Tensors & Autograd**  
  **Why**: Master core DL primitives.  
  **Tasks**: Rebuild Day 2 MLP in PyTorch.  
  **Proof**: Commit notebook.  
  **Level-up**: Compare parameter counts.

- **Day 30 – Manual Training Loop**  
  **Why**: Understand every operation.  
  **Tasks**: Train 3-layer MLP on Fashion-MNIST, 5 epochs.  
  **Proof**: Commit loss curve.  
  **Level-up**: Add validation split.

- **Day 31 – GPU Speed-Up**  
  **Why**: Leverage GPU for faster training.  
  **Tasks**: Move Day 30 model to GPU, compare runtime.  
  **Proof**: Commit runtime table.  
  **Level-up**: Profile with `torch.profiler`.

- **Day 32 – Optimizers Showdown**  
  **Why**: Compare SGD vs Adam.  
  **Tasks**: Train same net with both, plot losses.  
  **Proof**: Commit plot.  
  **Level-up**: Try RMSProp.

- **Day 33 – Regularization Tricks**  
  **Why**: Prevent overfitting.  
  **Tasks**: Add dropout (0.3) and weight decay (1e-4).  
  **Proof**: Commit validation accuracy table.  
  **Level-up**: Add BatchNorm.

- **Day 34 – Tiny CNN on CIFAR-10**  
  **Why**: Introduction to image models.  
  **Tasks**: Build 3-conv CNN, achieve ≥60% accuracy.  
  **Proof**: Commit accuracy.  
  **Level-up**: Plot confusion matrix.

- **Day 35 – Data Augmentation Gains**  
  **Why**: Boost accuracy with transforms.  
  **Tasks**: Add random crop/flip, reach ≥65% accuracy.  
  **Proof**: Commit accuracy curve.  
  **Level-up**: Try Cutout augmentation.

- **Day 36 – TensorBoard Logging**  
  **Why**: Visualize training dynamics.  
  **Tasks**: Log losses/images per epoch.  
  **Proof**: Commit TensorBoard screenshot.  
  **Level-up**: Log projection embeddings.

- **Day 37 – Transfer Learning MobileNet-V2**  
  **Why**: Leverage pretrained models.  
  **Tasks**: Fine-tune on Oxford Flowers (10-class subset), ≥80% accuracy.  
  **Proof**: Commit accuracy.  
  **Level-up**: Freeze encoder.

- **Day 38 – Mixed Precision AMP**  
  **Why**: Optimize speed and memory.  
  **Tasks**: Use `autocast` and `GradScaler`, compare step time.  
  **Proof**: Commit timing table.  
  **Level-up**: Try BF16 on CPU.

- **Day 39 – Checkpoints & Resume**  
  **Why**: Ensure resilience in long runs.  
  **Tasks**: Save best val accuracy, resume mid-epoch.  
  **Proof**: Commit console logs.  
  **Level-up**: Keep last 3 checkpoints.

- **Day 40 – PyTorch Lightning Intro**  
  **Why**: Reduce boilerplate code.  
  **Tasks**: Wrap Day 37 model as LightningModule.  
  **Proof**: Commit training logs.  
  **Level-up**: Use `trainer.predict`.

- **Day 41 – Callbacks Early Stop**  
  **Why**: Automate training decisions.  
  **Tasks**: Add EarlyStopping (patience=3) and ModelCheckpoint.  
  **Proof**: Commit callback logs.  
  **Level-up**: Add LearningRateMonitor.

- **Day 42 – Mini-Project: Dogs-vs-Cats32**  
  **Why**: Cap image-based learning.  
  **Tasks**: Achieve ≥85% accuracy on 32×32 subset.  
  **Proof**: Commit leaderboard screenshot.  
  **Level-up**: Create Grad-CAM demo GIF.

---

### Phase 4: Vision Extras (Days 43–50)
Explore advanced computer vision techniques, still T4-compatible.

- **Day 43 – Grad-CAM Heatmaps**  
  **Prep**: `pytorch-grad-cam`  
  **Tasks**: Overlay heatmap on Dog/Cat image.  
  **Proof**: Commit PNG.  
  **Level-up**: Process in batch mode.

- **Day 44 – YOLOv8-Nano Fine-Tune**  
  **Tasks**: Fine-tune 5 epochs on 10 custom images.  
  **Proof**: Commit mAP.  
  **Level-up**: Export to ONNX.

- **Day 45 – Tiny U-Net Segmentation**  
  **Tasks**: Train U-Net on small segmentation dataset.  
  **Proof**: Commit IoU metric.  
  **Level-up**: Visualize segmentation masks.

- **Day 46 – Label Studio Basics**  
  **Tasks**: Label 5 images, export COCO JSON.  
  **Proof**: Commit JSON file.  
  **Level-up**: Automate labeling pipeline.

- **Day 47 – ONNX Export**  
  **Tasks**: Export MobileNet-V2 to ONNX, run with `onnxruntime`.  
  **Proof**: Commit latency (ms).  
  **Level-up**: Optimize with ONNX opt level.

- **Day 48 – Int8 Quantization**  
  **Tasks**: Quantize MobileNet dynamically, compare Top-1 drop.  
  **Proof**: Commit accuracy table.  
  **Level-up**: Measure inference speedup.

- **Day 49 – Channel Pruning**  
  **Tasks**: Prune 30% of channels, fine-tune 1 epoch.  
  **Proof**: Commit model size reduction.  
  **Level-up**: Compare accuracy pre/post-pruning.

- **Day 50 – Mini-Project: Streamlit Grad-CAM App**  
  **Tasks**: Build app for image upload → heatmap output.  
  **Proof**: Commit deployed Streamlit URL.  
  **Level-up**: Add batch processing.

---

### Phase 5: NLP & Transformers (Days 51–70)
Dive into NLP with small, T4-friendly models (Distil* models, TinyBERT, TinyLlama 4-bit) and datasets (IMDb 5k, WikiText-2 10k).

- **Day 51 – Text Cleaning & Tokenization**  
  **Why**: Clean data for better models.  
  **Prep**: `import pandas as pd, re, nltk; nltk.download('stopwords')`  
  **Tasks**: Create `clean_text()`, process IMDb 5k reviews, save `imdb_clean.csv`.  
  **Proof**: Commit CSV and `clean_text.py`.  
  **Level-up**: Plot top-20 tokens.

- **Day 52 – Word2Vec Embeddings**  
  **Why**: Learn distributed representations.  
  **Prep**: `!pip install -q gensim`  
  **Tasks**: Train Word2Vec (50d) on IMDb, query similar words.  
  **Proof**: Commit screenshot and `word2vec.model`.  
  **Level-up**: Visualize embeddings with t-SNE.

- **Day 53 – GRU Sentiment Classifier**  
  **Why**: Explore recurrent nets.  
  **Prep**: `!pip install -q torchtext`  
  **Tasks**: Build GRU classifier, train 3 epochs, ≥75% accuracy.  
  **Proof**: Commit `day53_gru_sentiment.ipynb`.  
  **Level-up**: Add dropout (0.3).

- **Day 54 – Scaled-Dot Attention in NumPy**  
  **Why**: Understand Transformer mechanics.  
  **Prep**: `import numpy as np`  
  **Tasks**: Implement attention, verify vs PyTorch.  
  **Proof**: Commit script with diff < 1e-5.  
  **Level-up**: Add causal mask.

- **Day 55 – Tiny Transformer Encoder**  
  **Why**: Build Transformer blocks.  
  **Prep**: `import torch, torch.nn as nn`  
  **Tasks**: Create 2-block encoder (<1M params), forward dummy batch.  
  **Proof**: Commit param count and output shape.  
  **Level-up**: Train on synthetic copy-task.

- **Day 56 – Positional Encodings Visualized**  
  **Why**: Understand Transformer position info.  
  **Prep**: `import matplotlib.pyplot as plt, torch, math`  
  **Tasks**: Implement sinusoidal encodings, plot 4 dimensions.  
  **Proof**: Commit PNG plot.  
  **Level-up**: Compare with learnable encodings.

- **Day 57 – Pre-train Mini-BERT (4M Params)**  
  **Why**: Experience masked language modeling.  
  **Prep**: `!pip install -q transformers datasets`  
  **Tasks**: Train 4-layer BERT on WikiText-2, loss < 3.0.  
  **Proof**: Commit loss curve.  
  **Level-up**: Run mask-filling inference.

- **Day 58 – Fine-Tune BERT on SST-2**  
  **Why**: Leverage transfer learning.  
  **Prep**: `!pip install -q datasets transformers`  
  **Tasks**: Fine-tune BERT on SST-2, ≥80% accuracy.  
  **Proof**: Commit validation accuracy.  
  **Level-up**: Use differential learning rates.

- **Day 59 – Train BPE Tokenizer**  
  **Why**: Optimize tokenization.  
  **Prep**: `!pip install -q tokenizers datasets`  
  **Tasks**: Train BPE tokenizer (16k vocab) on IMDb+Wiki.  
  **Proof**: Commit `vocab.json`, `merges.txt`.  
  **Level-up**: Benchmark encoding speed.

- **Day 60 – DistilGPT-2 Jokes Fine-Tune**  
  **Why**: Build a generative model.  
  **Prep**: `!pip install -q transformers datasets`  
  **Tasks**: Fine-tune DistilGPT-2 on Reddit Jokes, generate 3 jokes.  
  **Proof**: Commit notebook and best joke screenshot.  
  **Level-up**: Use LoRA with `peft`.

- **Day 61 – Decoding Strategies**  
  **Why**: Control generation creativity.  
  **Tasks**: Implement greedy, top-k, nucleus sampling; evaluate diversity.  
  **Proof**: Commit diversity metrics and ratings.  
  **Level-up**: Add beam search.

- **Day 62 – LoRA Haiku Fine-Tune**  
  **Why**: Efficient fine-tuning on low VRAM.  
  **Prep**: `!pip install -q peft transformers datasets`  
  **Tasks**: Apply LoRA (rank 8) to DistilGPT-2, train on haikus.  
  **Proof**: Commit adapter file size and sample haiku.  
  **Level-up**: Merge LoRA weights.

- **Day 63 – PEFT vs Full Fine-Tune Eval**  
  **Why**: Quantify efficiency gains.  
  **Tasks**: Compare LoRA vs full fine-tune on perplexity and VRAM.  
  **Proof**: Commit table.  
  **Level-up**: Plot training time.

- **Day 64 – Build FAISS Index**  
  **Why**: Enable retrieval-augmented generation.  
  **Prep**: `!pip install -q faiss-cpu`  
  **Tasks**: Build FAISS index on Wikipedia embeddings, measure latency.  
  **Proof**: Commit index size and latency.  
  **Level-up**: Try IVF index.

- **Day 65 – LangChain Q&A Bot**  
  **Why**: Combine components for Q&A.  
  **Prep**: `!pip install -q langchain openai`  
  **Tasks**: Build RetrievalQA with FAISS and DistilGPT-2, answer 3 questions.  
  **Proof**: Commit answers screenshot.  
  **Level-up**: Add Streamlit UI.

- **Day 66 – Prompt Engineering Few-Shot**  
  **Why**: Optimize prompts for performance.  
  **Tasks**: Test zero/one/three-shot prompts on SST-2, compare accuracy.  
  **Proof**: Commit accuracy table.  
  **Level-up**: Use automatic prompt generation.

- **Day 67 – Chain-of-Thought Reasoning**  
  **Why**: Improve reasoning with CoT.  
  **Tasks**: Solve 10 math problems with/without CoT, compare accuracy.  
  **Proof**: Commit results.  
  **Level-up**: Try self-consistency decoding.

- **Day 68 – Add Classification Head**  
  **Why**: Repurpose pretrained models.  
  **Tasks**: Add head to mini-BERT, fine-tune on emotion dataset.  
  **Proof**: Commit macro-F1 score.  
  **Level-up**: Freeze encoder, compare.

- **Day 69 – Toxicity Filtering**  
  **Why**: Ensure safe outputs.  
  **Prep**: `!pip install -q detoxify`  
  **Tasks**: Filter 100 jokes, compute % removed.  
  **Proof**: Commit stats.  
  **Level-up**: Fine-tune on filtered jokes.

- **Day 70 – Mini-Project: Telegram Shakespeare Bot**  
  **Why**: Deploy a text model publicly.  
  **Prep**: `!pip install -q python-telegram-bot`  
  **Tasks**: Build RAG-based bot with Shakespeare style, host via `ngrok`.  
  **Proof**: Commit chat screencap and repo link.  
  **Level-up**: Deploy to Hugging Face Spaces.

---

### Phase 6: LLM Internals & Optimization (Days 71–85)
Dive into large language model mechanics and optimization techniques.

- **Day 71 – Self-Attention Math & KV Shapes**  
  **Why**: Derive attention mechanics.  
  **Prep**: `import torch, math`  
  **Tasks**: Derive Q,K,V shapes, compute attention manually, verify vs PyTorch.  
  **Proof**: Commit derivation and assertion pass.  
  **Level-up**: Show KV caching halves compute.

- **Day 72 – KV-Cache Timing**  
  **Why**: Measure caching benefits.  
  **Tasks**: Compare DistilGPT-2 generation with/without KV cache.  
  **Proof**: Commit tokens/s table.  
  **Level-up**: Plot speed vs sequence length.

- **Day 73 – Speculative Decoding**  
  **Why**: Accelerate sampling.  
  **Tasks**: Implement speculative decoding, measure ≥1.4× speedup.  
  **Proof**: Commit notebook.  
  **Level-up**: Vary acceptance threshold.

- **Day 74 – Gradient Checkpointing Demo**  
  **Why**: Save memory at compute cost.  
  **Prep**: `import torch.utils.checkpoint`  
  **Tasks**: Train CNN with/without checkpointing, measure VRAM.  
  **Proof**: Commit memory savings %.  
  **Level-up**: Compare epoch time.

- **Day 75 – torch.compile Benchmark**  
  **Why**: Leverage PyTorch compiler.  
  **Prep**: `!pip install -q torch==2.1`  
  **Tasks**: Benchmark TinyLlama with/without `torch.compile`.  
  **Proof**: Commit throughput table.  
  **Level-up**: Test different backends.

- **Day 76 – ZeRO Offload on CPU**  
  **Why**: Enable large models on single GPU.  
  **Prep**: `!pip install -q deepspeed`  
  **Tasks**: Train mini-BERT with ZeRO-3, log <3 GB GPU usage.  
  **Proof**: Commit notebook log.  
  **Level-up**: Compare speed vs FP16.

- **Day 77 – QLoRA TinyLlama**  
  **Why**: Efficient fine-tuning.  
  **Prep**: `!pip install -q peft bitsandbytes transformers accelerate`  
  **Tasks**: Fine-tune TinyLlama with QLoRA, <6 GB VRAM.  
  **Proof**: Commit loss curve.  
  **Level-up**: Export merged adapter.

- **Day 78 – Knowledge Distillation Student**  
  **Why**: Create smaller models.  
  **Tasks**: Distill DistilGPT-2 into 15M param student, compare perplexity.  
  **Proof**: Commit perplexity plot.  
  **Level-up**: Quantize student to Int8.

- **Day 79 – Toy Mixture-of-Experts**  
  **Why**: Explore efficient architectures.  
  **Prep**: `import torch.nn as nn`  
  **Tasks**: Build MoE with 2 experts, verify routing.  
  **Proof**: Commit routing histogram.  
  **Level-up**: Add load-balancing loss.

- **Day 80 – Sparse Attention Mask**  
  **Why**: Handle long sequences.  
  **Prep**: `import torch`  
  **Tasks**: Implement block-sparse mask, verify zeros.  
  **Proof**: Commit matrix visualization.  
  **Level-up**: Benchmark vs dense attention.

- **Day 81 – RLHF Bandit Summarizer**  
  **Why**: Learn preference optimization.  
  **Prep**: `!pip install -q trl datasets`  
  **Tasks**: Train DistilGPT-2 with PPO for summarization, show reward curve.  
  **Proof**: Commit rising reward curve.  
  **Level-up**: Try KL-penalized decoding.

- **Day 82 – KL Penalty Tuning**  
  **Why**: Stabilize RLHF training.  
  **Tasks**: Vary KL coefficient, plot rewards.  
  **Proof**: Commit chart.  
  **Level-up**: Find optimal coefficient.

- **Day 83 – Evaluation Harness**  
  **Why**: Automate model comparisons.  
  **Prep**: `!pip install -q evaluate nltk rouge_score`  
  **Tasks**: Build CLI for BLEU/ROUGE-L, run on Day 78 models.  
  **Proof**: Commit CLI output screenshot.  
  **Level-up**: Add WSC accuracy.

- **Day 84 – Model Card Documentation**  
  **Why**: Ensure transparency.  
  **Tasks**: Write model card with training data, metrics, limitations.  
  **Proof**: Commit `README.md`.  
  **Level-up**: Generate JSON-schema card.

- **Day 85 – Mini-Project: Publish to Hugging Face Hub**  
  **Why**: Share your work.  
  **Prep**: `!pip install -q huggingface_hub`  
  **Tasks**: Push student model and card, create Gradio demo Space.  
  **Proof**: Commit model repo and Space URLs.  
  **Level-up**: Add inference widget to README.

---

### Phase 7: Lightweight MLOps & Serving (Days 86–94)
Learn production-ready skills without heavy infrastructure.

- **Day 86 – Data Versioning with DVC**  
  **Why**: Ensure data reproducibility.  
  **Prep**: `!pip install -q dvc[s3]`  
  **Tasks**: Track `imdb_clean.csv` with DVC, push to GitHub.  
  **Proof**: Commit `.dvc` files diff.  
  **Level-up**: Add pre-commit hook.

- **Day 87 – CI Test & Slim Docker Build**  
  **Why**: Automate testing and deployment.  
  **Tasks**: Add `pytest` and GitHub Actions CI for tests and slim Docker build.  
  **Proof**: Commit green CI badge screenshot.  
  **Level-up**: Cache Docker layers.

- **Day 88 – Serve Model with TorchServe**  
  **Why**: Deploy production-grade REST APIs.  
  **Prep**: `!pip install -q torchserve torch-model-archiver`  
  **Tasks**: Archive DistilGPT-2, launch TorchServe, test with `curl`.  
  **Proof**: Commit latency and response example.  
  **Level-up**: Stress-test with concurrency.

- **Day 89 – FastAPI JWT Gateway**  
  **Why**: Add authentication and streaming.  
  **Prep**: `!pip install -q fastapi uvicorn python-multipart pyjwt`  
  **Tasks**: Build `/generate` endpoint with JWT, stream via SSE.  
  **Proof**: Commit screencast of `curl` with token.  
  **Level-up**: Add rate-limiting.

- **Day 90 – Resource Logging with psutil**  
  **Why**: Monitor system resources.  
  **Prep**: `import psutil, time, json`  
  **Tasks**: Log CPU/RAM/GPU usage during FastAPI calls, save `metrics.csv`.  
  **Proof**: Commit CSV and plot.  
  **Level-up**: Add Prometheus integration.

- **Day 91 – Traffic Splitter Canary**  
  **Why**: Enable safe model releases.  
  **Tasks**: Split 10% traffic to new model, compare latency.  
  **Proof**: Commit latency histogram.  
  **Level-up**: Wrap in Makefile.

- **Day 92 – Data Drift Detection**  
  **Why**: Detect model degradation.  
  **Prep**: `!pip install -q evidently`  
  **Tasks**: Run KS test on IMDb sentiment, save HTML report.  
  **Proof**: Commit HTML report.  
  **Level-up**: Add threshold alarm.

- **Day 93 – Offline A/B Analytics**  
  **Why**: Evaluate variants offline.  
  **Prep**: `import pandas as pd`  
  **Tasks**: Simulate 10k impressions, compute CTR, run chi-square test.  
  **Proof**: Commit conclusion markdown.  
  **Level-up**: Plot daily CTR.

- **Day 94 – Cost per 1K Tokens Notebook**  
  **Why**: Understand deployment costs.  
  **Prep**: `import pandas as pd`  
  **Tasks**: Compare provider costs, calculate $/1K tokens, plot bar chart.  
  **Proof**: Commit notebook and chart.  
  **Level-up**: Add sensitivity analysis.

---

### Phase 8: Capstone Sprint (Days 95–100)
Build and deploy a portfolio-ready AI product.

- **Day 95 – Ideation & Spec**  
  **Why**: Define clear project scope.  
  **Tasks**: Brainstorm 3 ideas, pick one with MoSCOW, write one-pager.  
  **Proof**: Commit `capstone_idea.md`.  
  **Level-up**: Revise based on peer feedback.

- **Day 96 – Repo Skeleton & Kanban**  
  **Why**: Organize project structure.  
  **Tasks**: Init repo with standard folders, set up GitHub Projects board.  
  **Proof**: Commit repo URL and board screenshot.  
  **Level-up**: Add Semantic-release config.

- **Day 97 – MVP Demo**  
  **Why**: Ship a functional prototype.  
  **Tasks**: Build minimal pipeline, record 2-min demo, open feedback issue.  
  **Proof**: Commit Loom link and issue.  
  **Level-up**: Add latency metrics to README.

- **Day 98 – Polish & Docs**  
  **Why**: Ensure professional quality.  
  **Tasks**: Add tests (≥60% coverage), Dockerfile, and mkdocs/Sphinx docs.  
  **Proof**: Commit CI badge and doc URL.  
  **Level-up**: Automate linting.

- **Day 99 – Public Launch**  
  **Why**: Share with real users.  
  **Tasks**: Deploy to Hugging Face Spaces/Render, post on X/LinkedIn.  
  **Proof**: Commit launch post links and URL.  
  **Level-up**: Submit to subreddit.

- **Day 100 – Retrospective & Next Steps**  
  **Why**: Reflect and plan growth.  
  **Tasks**: Write blog post, list 5 focus areas, thank contributors.  
  **Proof**: Commit blog post link.  
  **Level-up**: Submit conference talk proposal.

---

## Survival Checklist

- **Mount Google Drive**: Use `from google.colab import drive; drive.mount('/content/gdrive')` to persist data.
- **Handle OOM Errors**: Reduce batch size or sequence length; call `torch.cuda.empty_cache()` between cells.
- **Reduce batch & seq length** whenever you see `CUDA out of memory`.
- **GPU Queue Full**: Switch to CPU (`Runtime > Change runtime type > None`) for long jobs.
- **Monitor Resources**: Keep a second Colab tab with `!nvidia-smi -l 2` to track VRAM.
- **Stay Accountable**: Share progress on X, LinkedIn, or blogs to stay motivated.

---

## Getting Started

1. Clone this repository:  
   ```bash
   git clone https://github.com/adityakamat24/100-Days-of-AI
