# Mamba: A PyTorch Implementation from Scratch

![Project Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.0%2B-orange)

This repository is my personal journey into implementing the Mamba architecture from scratch, based on the paper "[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)". I'm building this to move beyond just using models and to really understand how they work from the ground up.

---

## My Goal for This Project

I wanted to tackle a recent, high-impact paper to challenge myself. My main goal is to build a solid, working implementation of the **Selective Scan (S6)** mechanism, which is the core innovation of Mamba. This project is my way of diving deep into structured state space models (SSMs) and figuring out why they're becoming a powerful alternative to Transformers.

---

## My Implementation Plan

Here's the roadmap I've set for myself. I'll be checking these off as I complete each part.

- [ ] **1. Discretization:** First, I need to figure out how to correctly implement the Zero-Order Hold method to discretize the continuous state space parameters (A, B).
- [ ] **2. Selective Scan (S6) Algorithm:** This is the big one. My goal is to build the parallel scan algorithm, which is what makes Mamba so efficient.
- [ ] **3. Mamba Block:** Once I have the core pieces, I'll assemble them into a single Mamba block, making sure the input-dependent selection logic is working correctly.
- [ ] **4. Full Mamba Model:** I'll stack the Mamba blocks to create the final language model architecture.
- [ ] **5. Training Script:** I'll write a clean script to train the model on a simple dataset like TinyShakespeare to make sure everything is learning as expected.
- [ ] **6. Inference Script:** Finally, I'll build a script to generate text and see my model in action.

---

## How to Follow Along

I'll be doing most of the core implementation in the `src/` directory. For more experimental code, rough notes, and visualizations, you can check out the `notebooks/` directory.

### Setup
If you want to run the code yourself, here’s how to get set up:
```bash
# Clone the repository
git clone [https://github.com/your-username/mamba-from-scratch.git](https://github.com/your-username/mamba-from-scratch.git)
cd mamba-from-scratch

# I recommend using a virtual environment
python -m venv venv
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt
```

---

## What I'm Learning Here
- **State Space Models (SSMs):** Getting a real feel for how SSMs work, not just reading about them.
- **The "Selective" Idea:** Digging into how Mamba uses the input to dynamically change its focus, which is a really cool concept.
- **Hardware-Aware Code:** Thinking about *why* the parallel scan is designed for GPUs and how that affects the implementation.
- **Beyond Quadratic Attention:** Moving past the Transformer's limitations by implementing a model with linear-time complexity.
