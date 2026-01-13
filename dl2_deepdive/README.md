### Deep Learning Research & Implementation Lab
This repository contains a collection of Jupyter Notebooks and Python modules designed to explore, implement, and visualize fundamental and advanced Deep Learning (DL) concepts. The materials cover the spectrum from basic perceptrons to state-of-the-art Transformer architectures and Reinforcement Learning.

#### Repository Structure 
**Core Utilities**
- dl_modules.py: It contains shared classes, custom PyTorch/Lightning wrappers, and utility functions (e.g., Classifier, Trainer, and evaluation metrics like BLEU) used across all notebooks to ensure code modularity and readability.

**Architectures & Mechanisms**
- dl_transformers.ipynb: Implementation of the Transformer Architecture. Explores multi-head self-attention, positional encoding, and the encoder-decoder structure.
- dl_attention_mechanism.ipynb: A deep dive into Queries, Keys, and Values. This notebook demonstrates how attention generates weighted representations of data.
- dl_multilayer_perceptrons.ipynb: Covers MLPs, activation functions (ReLU), backpropagation, and numerical stability.
- dl_classification.ipynb: Focuses on multi-class classification using Softmax Regression and loss function optimization.

**Computer Vision & NLP**
- dl_computer_vision.ipynb: Practical applications in CV, specifically focusing on Image Augmentation (flipping, color jittering) and training ResNet models on the CIFAR-10 dataset.
- dl_natural_language_processing.ipynb: Applications in NLP including Sentiment Analysis on the IMDb dataset and explorations of BERT and TextCNN.

**Advanced Learning Paradigms**
- dl_reinforcement_learning.ipynb: An introduction to Reinforcement Learning. Covers Markov Decision Processes (MDP), Value Iteration, and Q-Learning using the FrozenLake environment.
- dl_gaussian_processes.ipynb: Bayesian approach to modeling, focusing on GP priors, posterior inference, and predictive confidence regions.

**Optimization & Scaling**
- dl_parallelism.ipynb: Techniques for Multi-GPU Training, including data sharding, gradient synchronization (All-Reduce), and hardware acceleration.


**Prerequisites** 
To run these notebooks, you will need: Python 3.8+; PyTorch & Torchvision; PyTorch Lightning; GPyTorch (for Gaussian Processes); Matplotlib & NumPy

Installation: Clone this repository.
Ensure dl_modules.py is in the same directory as the notebooks, as they rely on it for imports.

It is recommended to use a GPU-enabled environment (like Google Colab or a local CUDA setup) for the dl_parallelism and dl_transformers notebooks.

**Acknowledgments**
The implementations and theoretical summaries in this repository are based on and modified from the Dive into Deep Learning (d2l.ai) open-source textbook.