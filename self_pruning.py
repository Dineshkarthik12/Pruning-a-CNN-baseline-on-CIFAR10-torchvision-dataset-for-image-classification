"""
Self-Pruning Neural Network for CIFAR-10
=========================================
Implements a PrunableLinear layer with learnable gate parameters that allow
the network to dynamically prune unimportant weights during training via
L1 sparsity regularisation on the sigmoid of the gate scores.

Architecture: CNN feature extractor + PrunableLinear classifier layers.

Usage:
    python self_pruning.py
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# ---------------------------------------------------------------------------
# 1. PrunableLinear Layer
# ---------------------------------------------------------------------------

class PrunableLinear(nn.Module):
    """Fully-connected layer whose weights are element-wise gated by learnable
    gate scores.  During the forward pass the effective weight matrix is
        W_eff = sigmoid(gate_scores) * weight
    By penalising the L1 norm of sigmoid(gate_scores) during training, the
    network learns to push unimportant gates towards zero, effectively pruning
    those connections.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight & bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Gate scores - one per weight element, initialised so that
        # sigmoid(gate_scores) starts at 0.5 (neutral starting point;
        # easier for the L1 penalty to push unimportant gates towards zero).
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
        # Initialise gates at 0.0 so sigmoid = 0.5 (neutral starting point;
        # easier for the L1 penalty to push unimportant gates towards zero).
        nn.init.constant_(self.gate_scores, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        effective_weight = torch.sigmoid(self.gate_scores) * self.weight
        return nn.functional.linear(x, effective_weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ---------------------------------------------------------------------------
# 2. Sparsity helpers
# ---------------------------------------------------------------------------

def collect_gate_values(model: nn.Module) -> torch.Tensor:
    """Return a 1-D tensor of all sigmoid(gate_scores) across the model."""
    gates = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates.append(torch.sigmoid(m.gate_scores).view(-1))
    return torch.cat(gates)


def sparsity_loss(model: nn.Module) -> torch.Tensor:
    """L1 norm (sum) of all sigmoid gate values - acts as the sparsity penalty.
    Using sum ensures strong gradient signal on each individual gate."""
    return collect_gate_values(model).sum()


def sparsity_percentage(model: nn.Module, threshold: float = 1e-2) -> float:
    """Percentage of gates whose sigmoid value is below *threshold*."""
    gates = collect_gate_values(model).detach()
    return (gates < threshold).float().mean().item() * 100.0


# ---------------------------------------------------------------------------
# 3. Model - CNN + PrunableLinear Classifier
# ---------------------------------------------------------------------------

class SelfPruningCNN(nn.Module):
    """CNN feature extractor followed by PrunableLinear classifier layers.

    Feature Extractor (fixed, no pruning):
        Conv2d(3,64)  -> BN -> ReLU -> Conv2d(64,128) -> BN -> ReLU -> MaxPool
        Conv2d(128,256) -> BN -> ReLU -> MaxPool
        -> AdaptiveAvgPool -> flatten to 256-dim

    Classifier (prunable):
        PrunableLinear(256, 128) -> BN -> ReLU -> Dropout
        PrunableLinear(128, 10)
    """

    def __init__(self):
        super().__init__()

        # --- Feature Extractor (Conv layers - NOT pruned) ---
        self.features = nn.Sequential(OrderedDict([
            ("conv1",   nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ("bn1",     nn.BatchNorm2d(64)),
            ("relu1",   nn.ReLU(inplace=True)),

            ("conv2",   nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ("bn2",     nn.BatchNorm2d(128)),
            ("relu2",   nn.ReLU(inplace=True)),
            ("pool1",   nn.MaxPool2d(2, 2)),                     # 32x32 -> 16x16

            ("conv3",   nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            ("bn3",     nn.BatchNorm2d(256)),
            ("relu3",   nn.ReLU(inplace=True)),
            ("pool2",   nn.MaxPool2d(2, 2)),                     # 16x16 -> 8x8

            ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),           # 8x8 -> 1x1
        ]))

        # --- Classifier (PrunableLinear layers - PRUNED) ---
        self.classifier = nn.Sequential(OrderedDict([
            ("fc1",     PrunableLinear(256, 128)),
            ("bn_fc1",  nn.BatchNorm1d(128)),
            ("relu_fc", nn.ReLU(inplace=True)),
            ("dropout", nn.Dropout(0.3)),
            ("fc2",     PrunableLinear(128, 10)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)          # flatten: (batch, 256)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# 4. Data loaders
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size: int = 128, data_dir: str = "./data"):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader


# ---------------------------------------------------------------------------
# 5. Train & evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, lam, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        cls_loss = criterion(outputs, targets)
        sp_loss = sparsity_loss(model)
        loss = cls_loss + lam * sp_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------

def plot_gate_distributions(all_gates: dict, save_path: str = "gate_distribution.png"):
    """Plot histograms of sigmoid(gate_scores) for each lambda value."""
    n = len(all_gates)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes.flatten()

    for ax, (lam_label, gates_np) in zip(axes, all_gates.items()):
        ax.hist(gates_np, bins=50, color="#4C72B0", edgecolor="white",
                alpha=0.85)
        ax.set_title(lam_label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Sigmoid(gate score)")
        ax.set_ylabel("Count")
        ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1,
                   label="Prune threshold (0.01)")
        ax.legend(fontsize=8)

    fig.suptitle("Distribution of Gate Values After Training",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Gate distribution plot saved -> {save_path}", flush=True)


# ---------------------------------------------------------------------------
# 7. Main experiment
# ---------------------------------------------------------------------------

def main():
    # ---- Configuration ----
    NUM_EPOCHS = 20
    BATCH_SIZE = 128
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results.json")
    PLOT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gate_distribution.png")

    # Lambda values: None, Low, Medium, High, Very High
    LAMBDA_CONFIG = [
        ("None (0.0)",       0.0),
        ("Low (1e-4)",       1e-4),
        ("Medium (1e-3)",    1e-3),
        ("High (1e-2)",      1e-2),
        ("Very High (5e-2)", 5e-2),
    ]

    print(f"[INFO] Device: {DEVICE}", flush=True)
    print(f"[INFO] Epochs: {NUM_EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LR}", flush=True)
    print(f"[INFO] Lambda levels: {[label for label, _ in LAMBDA_CONFIG]}", flush=True)
    print("=" * 70, flush=True)

    trainloader, testloader = get_dataloaders(BATCH_SIZE, DATA_DIR)

    results = []          # list of dicts for the report
    all_gates = {}        # label -> numpy array of gate values

    for label, lam in LAMBDA_CONFIG:
        print(f"\n{'-' * 70}", flush=True)
        print(f"  Training with {label}", flush=True)
        print(f"{'-' * 70}", flush=True)

        model = SelfPruningCNN().to(DEVICE)
        
        gate_params = []
        base_params = []
        for name, param in model.named_parameters():
            if "gate_scores" in name:
                gate_params.append(param)
            else:
                base_params.append(param)

        optimizer = optim.Adam([
            {"params": base_params},
            {"params": gate_params, "lr": LR * 50.0} 
        ], lr=LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=NUM_EPOCHS)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, trainloader, optimizer, criterion, lam, DEVICE)
            test_acc = evaluate(model, testloader, DEVICE)
            sp = sparsity_percentage(model)
            scheduler.step()

            print(f"  Epoch {epoch:>2}/{NUM_EPOCHS}  "
                  f"Loss: {train_loss:.4f}  "
                  f"Train Acc: {train_acc:.2f}%  "
                  f"Test Acc: {test_acc:.2f}%  "
                  f"Sparsity: {sp:.2f}%", flush=True)

        # Final evaluation
        final_test_acc = evaluate(model, testloader, DEVICE)
        final_sparsity = sparsity_percentage(model)
        gate_vals = collect_gate_values(model).detach().cpu().numpy()

        print(f"\n  > Final  Test Acc: {final_test_acc:.2f}%  "
              f"Sparsity: {final_sparsity:.2f}%", flush=True)

        results.append({
            "lambda_label": label,
            "lambda": lam,
            "test_accuracy": round(final_test_acc, 2),
            "sparsity_pct": round(final_sparsity, 2),
        })
        all_gates[label] = gate_vals

    # ---- Save results & plots ----
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved -> {RESULTS_FILE}", flush=True)

    plot_gate_distributions(all_gates, PLOT_FILE)

    # ---- Pretty-print summary ----
    print("\n" + "=" * 70, flush=True)
    print(f"{'Lambda Level':<22} | {'Test Accuracy':>14} | {'Sparsity (%)':>14}", flush=True)
    print("-" * 56, flush=True)
    for r in results:
        print(f"{r['lambda_label']:<22} | {r['test_accuracy']:>13.2f}% | "
              f"{r['sparsity_pct']:>13.2f}%", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
