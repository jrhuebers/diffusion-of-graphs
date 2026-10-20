"""A tiny DDPM for fixed-size, undirected graphs represented by edge vectors.

Train on synthetic stochastic-block-model graphs, then sample new graphs:
    python graph_diffusion.py --train --steps 1000
    python graph_diffusion.py --sample --checkpoint graph_ddpm.pt --num-samples 4
"""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Denoiser(nn.Module):
    """Predict the Gaussian noise in a noisy graph at diffusion step t."""

    def __init__(self, edge_count, hidden=256, time_dim=32):
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(edge_count + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, edge_count),
        )

    def forward(self, x, t):
        half = self.time_dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=x.device) * -scale)
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        time = torch.cat((angles.sin(), angles.cos()), dim=1)
        return self.net(torch.cat((x, time), dim=1))


def make_sbm_graphs(count, nodes, device="cpu"):
    """Generate simple two-community undirected graphs with varied densities."""
    i, j = torch.triu_indices(nodes, nodes, offset=1, device=device)
    graphs = torch.empty(count, i.numel(), device=device)
    for k in range(count):
        labels = torch.randint(0, 2, (nodes,), device=device)
        same = labels[i] == labels[j]
        probs = torch.where(same, 0.35, 0.08)
        graphs[k] = (torch.rand(i.numel(), device=device) < probs).float()
    return graphs * 2 - 1, (i, j)


def schedule(steps, device):
    betas = torch.linspace(1e-4, 0.02, steps, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, _ = make_sbm_graphs(args.dataset_size, args.nodes)
    loader = DataLoader(TensorDataset(data), batch_size=args.batch_size, shuffle=True)
    model = Denoiser(data.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    _, _, alpha_bars = schedule(args.diffusion_steps, device)
    model.train()
    for step in range(1, args.steps + 1):
        epoch_loss = 0.0
        for (x,) in loader:
            x = x.to(device)
            t = torch.randint(args.diffusion_steps, (x.size(0),), device=device)
            noise = torch.randn_like(x)
            a_bar = alpha_bars[t].unsqueeze(1)
            noisy = a_bar.sqrt() * x + (1 - a_bar).sqrt() * noise
            loss = nn.functional.mse_loss(model(noisy, t), noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            print(f"epoch {step:4d}/{args.steps}  loss {epoch_loss / len(loader):.4f}")
    torch.save({"model": model.cpu().state_dict(), "nodes": args.nodes,
                "diffusion_steps": args.diffusion_steps}, args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")


@torch.no_grad()
def sample(checkpoint, count, output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    nodes, steps = saved["nodes"], saved["diffusion_steps"]
    i, j = torch.triu_indices(nodes, nodes, offset=1, device=device)
    model = Denoiser(i.numel()).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    betas, alphas, alpha_bars = schedule(steps, device)
    x = torch.randn(count, i.numel(), device=device)
    for step in reversed(range(steps)):
        t = torch.full((count,), step, device=device, dtype=torch.long)
        beta, alpha, a_bar = betas[step], alphas[step], alpha_bars[step]
        prediction = model(x, t)
        x = (x - beta / torch.sqrt(1 - a_bar) * prediction) / torch.sqrt(alpha)
        if step:
            x += beta.sqrt() * torch.randn_like(x)
    adjacency = torch.zeros(count, nodes, nodes, device=device)
    edges = (x > 0).float()
    adjacency[:, i, j] = edges
    adjacency[:, j, i] = edges
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adjacency.cpu(), output)
    for k, graph in enumerate(adjacency.cpu()):
        print(f"sample {k}: {int(graph.sum().item() / 2)} edges")
    print(f"Saved adjacency matrices [samples, nodes, nodes] to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="train on synthetic SBM graphs")
    mode.add_argument("--sample", action="store_true", help="sample graphs from checkpoint")
    parser.add_argument("--nodes", type=int, default=16, help="fixed node count")
    parser.add_argument("--steps", type=int, default=1000, help="training epochs")
    parser.add_argument("--diffusion-steps", type=int, default=100, help="DDPM noise steps")
    parser.add_argument("--dataset-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default="graph_ddpm.pt")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--output", default="samples.pt")
    args = parser.parse_args()
    if args.nodes < 2 or args.steps < 1 or args.diffusion_steps < 2:
        parser.error("nodes >= 2, steps >= 1, and diffusion-steps >= 2 are required")
    train(args) if args.train else sample(args.checkpoint, args.num_samples, args.output)


if __name__ == "__main__":
    main()
