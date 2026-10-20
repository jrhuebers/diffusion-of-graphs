# Tiny Graph Diffusion

A minimal **Gaussian DDPM** that generates fixed-size, undirected graphs. Each graph is encoded as its upper-triangular edge vector (`-1` = no edge, `+1` = edge); a small MLP learns to reverse the noise process. The included toy training data is generated from two-community stochastic block models (SBMs), so no dataset download is needed.

## Quick start

```bash
python -m pip install -r requirements.txt
python graph_diffusion.py --train --steps 1000
python graph_diffusion.py --sample --checkpoint graph_ddpm.pt --num-samples 4
```

Training saves `graph_ddpm.pt`. Sampling saves a PyTorch tensor of binary symmetric adjacency matrices, shape `[num_samples, nodes, nodes]`, to `samples.pt`. Nodes are fixed per checkpoint and there are no self-loops. CUDA is used automatically when available.

Useful options: `--nodes 16`, `--dataset-size 2048`, `--batch-size 128`, `--diffusion-steps 100`, `--checkpoint graph_ddpm.pt`, `--output samples.pt`.

This is a compact educational baseline, not a state-of-the-art or discrete graph diffusion method. It learns only the synthetic SBM distribution; swap out `make_sbm_graphs` to train on your own fixed-size graph dataset.
