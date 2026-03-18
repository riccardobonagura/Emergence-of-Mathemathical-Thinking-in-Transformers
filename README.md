## Emergence of Mathematical Reasoning in Transformers:
  Geometric Analysis of Representation Spaces and Fine-Tuning Dynamics
  Bachelor’s Thesis in Computer Engineering, University of Naples Federico II
  Riccardo Bonagura, N46007216

## Overview
Modern Large Language Models exhibit surprising capabilities in logical-mathematical reasoning, yet the internal mechanisms through which these abilities emerge remain largely opaque. This work aims to investigate the geometric structure of the internal representation spaces of Transformers—namely, the hidden states—in order to understand where and how mathematical reasoning takes shape within the layered hierarchy of a language model.
The project is structured in three distinct phases: a descriptive phase, a decoding phase, and a dynamic phase.

## Research Questions
Three related questions:
1. Within the Transformer hierarchy, is there a specific layer beyond which the internal representations of mathematical entities acquire a geometric structure distinguishable from that of generic language?
2. Are specific mathematical properties (such as parity, sign, order relations, or operator type) linearly decodable?
3. If model performance on mathematical reasoning benchmarks improves through a fine-tuning cycle, does this improvement correspond to a measurable reorganization of the geometry of internal embeddings? If so, in which layers is this change concentrated?

## Project Pipeline

### Dataset construction
A dataset of stimuli divided into four categories: elementary arithmetic, algebraic expressions, mathematical word problems (from GSM8K), and generic control text.
The stimuli will be minimally contrastive, i.e., pairs differing by only one element at a time (e.g., [5 * 3 = ?] vs [5 + 3 = ?]). This ensures that observed geometric differences can be attributed to mathematical structure rather than confounding variables.

### Descriptive phase
Hidden states are extracted for each layer using TransformerLens and analyzed through three metrics:
Isotropy per layer and input category, measured as the average cosine similarity between random vector pairs;
Intra-model Centered Kernel Alignment (CKA), producing a layer × layer heatmap of representational similarity;
UMAP projections of hidden states to visualize geometric separation between categories.

### Decoding phase
For each layer, linear probing classifiers (simple logistic regressions) are trained on specific mathematical targets, producing the central curve of the study: probing accuracy as a function of layer depth.
The weights of these classifiers are then interpreted as linear directions in hidden-state space, in search of an interpretable geometry for properties such as sign, parity, and operator type.

### Dynamic phase
A fine-tuning cycle using QLoRA on the MetaMath dataset is performed with regular checkpointing.
The geometric metrics developed in previous phases are recomputed at each checkpoint, reconstructing the model’s geometric trajectory during learning and comparing it with the performance delta measured on GSM8K.


## Mathematical Tools
Applied linear algebra: dot products, cosine similarity, PCA decomposition, Gram matrix algebra
Statistics: logistic regression, analysis of variance across layers, measures of linear separability
Centered Kernel Alignment is formulated as a normalized inner product between centered Gram matrices, and its interpretation as an invariance measure leverages the concept of equivalence up to orthogonal transformations.
Embedding isotropy relates to the theory of random vector distributions in high-dimensional spaces and the phenomenon of measure concentration.

## Software
- TransformerLens for hidden state extraction and activation hooks
- HuggingFace Transformers and PEFT for QLoRA fine-tuning
- bitsandbytes for 4-bit quantization
- scikit-learn for probing classifiers
- umap-learn for dimensionality reduction
- NumPy and PyTorch for custom metrics
- Weights & Biases for training logging and monitoring geometric metrics across checkpoints
- Plotly and Seaborn for visualization

## Candidate Models
Open-source Small Language Models runnable on consumer hardware (notably a 16GB NVIDIA GPU):
GPT-2 medium (355M parameters, OpenAI), used for development and code validation due to extensive interpretability literature
Phi-3-mini-4k-instruct (3.8B parameters, Microsoft), time permitting, used for final experiments due to its stronger mathematical reasoning capabilities and clearer geometric signals

## Deliverables
This thesis project will produce:
1. A Python codebase
2. A stimulus dataset
3. Visualizations: layer-wise probing curves, CKA heatmaps, comparative UMAP projections, geometric trajectories across checkpoints
4. A written report discussing the results and providing a coherent resolution of the three research questions

