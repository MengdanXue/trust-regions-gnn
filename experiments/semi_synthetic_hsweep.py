#!/usr/bin/env python3
"""
Semi-Synthetic H-Sweep Experiment
==================================

关键实验：用真实数据的节点特征，通过Edge Rewiring改变图的同质性h
验证U-Shape是否在"真实特征"上也成立

这是冲TKDE的关键实验！

Author: FSD Framework
Date: 2025-12-23
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


@dataclass
class SemiSyntheticResult:
    """Single experiment result"""
    dataset: str
    target_h: float
    actual_h: float
    original_h: float
    gcn_acc: float
    gat_acc: float
    sage_acc: float
    mlp_acc: float
    gcn_std: float
    gat_std: float
    sage_std: float
    mlp_std: float
    best_model: str
    gcn_advantage: float
    n_nodes: int
    n_edges: int


# ============== Models ==============

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


# ============== Edge Rewiring ==============

def compute_homophily(edge_index: torch.Tensor, labels: torch.Tensor) -> float:
    """计算边同质性"""
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()


def rewire_edges_to_target_h(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    target_h: float,
    n_nodes: int,
    max_iterations: int = 10000
) -> torch.Tensor:
    """
    通过Edge Rewiring将图的同质性调整到目标值

    关键思想：
    - 如果当前h > target_h，需要增加异质边（连接不同类别）
    - 如果当前h < target_h，需要增加同质边（连接相同类别）
    """
    # 转换为可操作的边集合
    edges = set()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src < dst:  # 避免重复
            edges.add((src, dst))

    n_edges = len(edges)
    labels_np = labels.cpu().numpy()

    # 按类别分组节点
    unique_labels = np.unique(labels_np)
    nodes_by_label = {l: np.where(labels_np == l)[0] for l in unique_labels}

    current_h = compute_homophily(edge_index, labels)

    for iteration in range(max_iterations):
        # 检查是否达到目标
        if abs(current_h - target_h) < 0.01:
            break

        # 随机选择一条边删除
        edge_list = list(edges)
        remove_idx = np.random.randint(len(edge_list))
        old_edge = edge_list[remove_idx]
        old_src, old_dst = old_edge

        # 决定新边的类型
        if current_h > target_h:
            # 需要降低h，添加异质边
            src_label = labels_np[old_src]
            # 选择不同类别的目标节点
            other_labels = [l for l in unique_labels if l != src_label]
            if other_labels:
                target_label = np.random.choice(other_labels)
                candidates = nodes_by_label[target_label]
                if len(candidates) > 0:
                    new_dst = np.random.choice(candidates)
                    new_src = old_src
                else:
                    continue
            else:
                continue
        else:
            # 需要提高h，添加同质边
            src_label = labels_np[old_src]
            candidates = nodes_by_label[src_label]
            candidates = [c for c in candidates if c != old_src]
            if len(candidates) > 0:
                new_dst = np.random.choice(candidates)
                new_src = old_src
            else:
                continue

        # 确保新边合法
        new_edge = (min(new_src, new_dst), max(new_src, new_dst))
        if new_edge not in edges and new_edge[0] != new_edge[1]:
            # 执行替换
            edges.remove(old_edge)
            edges.add(new_edge)

            # 重新计算h
            edge_list = list(edges)
            new_edge_index = torch.tensor(
                [[e[0] for e in edge_list] + [e[1] for e in edge_list],
                 [e[1] for e in edge_list] + [e[0] for e in edge_list]],
                dtype=torch.long
            )
            current_h = compute_homophily(new_edge_index, labels)

    # 构建最终的edge_index
    edge_list = list(edges)
    final_edge_index = torch.tensor(
        [[e[0] for e in edge_list] + [e[1] for e in edge_list],
         [e[1] for e in edge_list] + [e[0] for e in edge_list]],
        dtype=torch.long
    )

    return final_edge_index


# ============== Training ==============

def train_and_evaluate(
    model: nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01
) -> float:
    """训练模型并返回测试准确率"""
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_acc = 0
    best_test_acc = 0
    patience = 30
    no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)

            val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

    return best_test_acc


# ============== Data Loading ==============

def load_cora():
    """加载Cora数据集"""
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    return data, 'Cora'


def load_citeseer():
    """加载CiteSeer数据集"""
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='./data', name='CiteSeer')
    data = dataset[0]
    return data, 'CiteSeer'


def load_pubmed():
    """加载PubMed数据集"""
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='./data', name='PubMed')
    data = dataset[0]
    return data, 'PubMed'


def load_elliptic():
    """加载Elliptic数据集"""
    data_path = Path(__file__).parent / "data" / "elliptic_weber_split.pkl"
    if not data_path.exists():
        print(f"Elliptic dataset not found at {data_path}")
        return None, None

    with open(data_path, 'rb') as f:
        loaded = pickle.load(f)

    if 'data' in loaded:
        data = loaded['data']
    else:
        data = loaded

    return data, 'Elliptic'


# ============== Main Experiment ==============

def run_semi_synthetic_hsweep(
    data: Data,
    dataset_name: str,
    target_h_values: List[float],
    n_runs: int = 5
) -> List[SemiSyntheticResult]:
    """
    在单个数据集上运行半合成H-Sweep
    """
    results = []

    # 原始同质性
    original_h = compute_homophily(data.edge_index, data.y)
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Original homophily: {original_h:.3f}")
    print(f"Nodes: {data.x.shape[0]}, Features: {data.x.shape[1]}")
    print(f"Edges: {data.edge_index.shape[1] // 2}")
    print(f"Classes: {len(torch.unique(data.y))}")
    print(f"{'='*70}")

    # 创建固定的train/val/test split
    n_nodes = data.x.shape[0]

    for target_h in target_h_values:
        print(f"\n--- Target h = {target_h:.2f} ---")

        gcn_accs, gat_accs, sage_accs, mlp_accs = [], [], [], []
        actual_hs = []

        for run in range(n_runs):
            seed = 42 + run * 100
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Edge Rewiring
            print(f"  Run {run+1}/{n_runs}: Rewiring edges...", end=" ")
            new_edge_index = rewire_edges_to_target_h(
                data.edge_index, data.y, target_h, n_nodes
            )
            actual_h = compute_homophily(new_edge_index, data.y)
            actual_hs.append(actual_h)
            print(f"h={actual_h:.3f}", end=" ")

            # 创建新的Data对象
            new_data = Data(
                x=data.x,
                edge_index=new_edge_index,
                y=data.y
            )

            # 随机划分
            perm = torch.randperm(n_nodes)
            train_idx = int(0.6 * n_nodes)
            val_idx = int(0.8 * n_nodes)

            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)

            train_mask[perm[:train_idx]] = True
            val_mask[perm[train_idx:val_idx]] = True
            test_mask[perm[val_idx:]] = True

            n_features = data.x.shape[1]
            n_classes = len(torch.unique(data.y))
            hidden = 64

            # Train MLP
            torch.manual_seed(seed)
            mlp = MLP(n_features, hidden, n_classes)
            mlp_acc = train_and_evaluate(mlp, new_data, train_mask, val_mask, test_mask)
            mlp_accs.append(mlp_acc)

            # Train GCN
            torch.manual_seed(seed)
            gcn = GCN(n_features, hidden, n_classes)
            gcn_acc = train_and_evaluate(gcn, new_data, train_mask, val_mask, test_mask)
            gcn_accs.append(gcn_acc)

            # Train GAT
            torch.manual_seed(seed)
            gat = GAT(n_features, hidden, n_classes)
            gat_acc = train_and_evaluate(gat, new_data, train_mask, val_mask, test_mask)
            gat_accs.append(gat_acc)

            # Train GraphSAGE
            torch.manual_seed(seed)
            sage = GraphSAGE(n_features, hidden, n_classes)
            sage_acc = train_and_evaluate(sage, new_data, train_mask, val_mask, test_mask)
            sage_accs.append(sage_acc)

            print(f"GCN={gcn_acc:.3f} GAT={gat_acc:.3f} SAGE={sage_acc:.3f} MLP={mlp_acc:.3f}")

        # 计算统计量
        gcn_mean, gcn_std = np.mean(gcn_accs), np.std(gcn_accs)
        gat_mean, gat_std = np.mean(gat_accs), np.std(gat_accs)
        sage_mean, sage_std = np.mean(sage_accs), np.std(sage_accs)
        mlp_mean, mlp_std = np.mean(mlp_accs), np.std(mlp_accs)
        actual_h_mean = np.mean(actual_hs)

        # 确定最佳模型
        model_accs = {'GCN': gcn_mean, 'GAT': gat_mean, 'SAGE': sage_mean, 'MLP': mlp_mean}
        best_model = max(model_accs, key=model_accs.get)

        result = SemiSyntheticResult(
            dataset=dataset_name,
            target_h=target_h,
            actual_h=actual_h_mean,
            original_h=original_h,
            gcn_acc=gcn_mean,
            gat_acc=gat_mean,
            sage_acc=sage_mean,
            mlp_acc=mlp_mean,
            gcn_std=gcn_std,
            gat_std=gat_std,
            sage_std=sage_std,
            mlp_std=mlp_std,
            best_model=best_model,
            gcn_advantage=gcn_mean - mlp_mean,
            n_nodes=n_nodes,
            n_edges=new_edge_index.shape[1] // 2
        )
        results.append(result)

        print(f"\n  Summary: GCN={gcn_mean:.3f}±{gcn_std:.3f}, "
              f"MLP={mlp_mean:.3f}±{mlp_std:.3f}, "
              f"GCN advantage={gcn_mean-mlp_mean:+.3f}, Best={best_model}")

    return results


def analyze_and_visualize(results: List[SemiSyntheticResult], dataset_name: str):
    """分析和可视化结果"""
    print(f"\n{'='*80}")
    print(f"SEMI-SYNTHETIC H-SWEEP RESULTS: {dataset_name}")
    print(f"{'='*80}")

    print(f"\n{'h':>6} {'h_act':>6} {'GCN':>8} {'GAT':>8} {'SAGE':>8} {'MLP':>8} {'GCN-MLP':>8} {'Best':>6}")
    print("-" * 70)

    for r in results:
        print(f"{r.target_h:>6.2f} {r.actual_h:>6.3f} {r.gcn_acc:>8.3f} {r.gat_acc:>8.3f} "
              f"{r.sage_acc:>8.3f} {r.mlp_acc:>8.3f} {r.gcn_advantage:>+8.3f} {r.best_model:>6}")

    # U-Shape分析
    print(f"\n{'='*80}")
    print("U-SHAPE ANALYSIS")
    print(f"{'='*80}")

    # 找到谷底
    gcn_advantages = [r.gcn_advantage for r in results]
    min_idx = np.argmin(gcn_advantages)
    valley_h = results[min_idx].actual_h
    valley_adv = results[min_idx].gcn_advantage

    print(f"\nGCN Advantage Valley:")
    print(f"  h = {valley_h:.3f}, GCN-MLP = {valley_adv:+.3f}")

    # 两端
    low_h_results = [r for r in results if r.actual_h < 0.3]
    high_h_results = [r for r in results if r.actual_h > 0.7]
    mid_h_results = [r for r in results if 0.3 <= r.actual_h <= 0.7]

    if low_h_results:
        low_h_adv = np.mean([r.gcn_advantage for r in low_h_results])
        print(f"\nLow-h zone (h < 0.3):")
        print(f"  Average GCN advantage: {low_h_adv:+.3f}")

    if mid_h_results:
        mid_h_adv = np.mean([r.gcn_advantage for r in mid_h_results])
        print(f"\nMid-h zone (0.3 <= h <= 0.7):")
        print(f"  Average GCN advantage: {mid_h_adv:+.3f}")

    if high_h_results:
        high_h_adv = np.mean([r.gcn_advantage for r in high_h_results])
        print(f"\nHigh-h zone (h > 0.7):")
        print(f"  Average GCN advantage: {high_h_adv:+.3f}")

    # 判断U-Shape是否存在
    print(f"\n{'='*80}")
    print("U-SHAPE VERDICT")
    print(f"{'='*80}")

    if low_h_results and mid_h_results and high_h_results:
        low_adv = np.mean([r.gcn_advantage for r in low_h_results])
        mid_adv = np.mean([r.gcn_advantage for r in mid_h_results])
        high_adv = np.mean([r.gcn_advantage for r in high_h_results])

        # U-Shape条件：两端比中间高
        is_u_shape = (low_adv > mid_adv) and (high_adv > mid_adv)

        if is_u_shape:
            u_depth = ((low_adv + high_adv) / 2) - mid_adv
            print(f"\n*** U-SHAPE CONFIRMED! ***")
            print(f"  Low-h advantage: {low_adv:+.3f}")
            print(f"  Mid-h advantage: {mid_adv:+.3f}")
            print(f"  High-h advantage: {high_adv:+.3f}")
            print(f"  U-Shape depth: {u_depth:.3f}")
        else:
            print(f"\n  U-Shape NOT confirmed on {dataset_name}")
            print(f"  Low: {low_adv:+.3f}, Mid: {mid_adv:+.3f}, High: {high_adv:+.3f}")
    else:
        print("\n  Insufficient data points for U-Shape analysis")

    return results


def main():
    """主函数"""
    print("="*80)
    print("SEMI-SYNTHETIC H-SWEEP EXPERIMENT")
    print("Key experiment for TKDE: Validate U-Shape on REAL FEATURES")
    print("="*80)

    # H值范围
    target_h_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_runs = 5

    all_results = {}

    # 尝试加载数据集
    datasets_to_try = [
        ('Cora', load_cora),
        ('CiteSeer', load_citeseer),
        # ('Elliptic', load_elliptic),  # 如果有的话
    ]

    for name, loader in datasets_to_try:
        try:
            print(f"\nLoading {name}...")
            data, dataset_name = loader()
            if data is None:
                print(f"  Failed to load {name}")
                continue

            # 运行实验
            results = run_semi_synthetic_hsweep(
                data, dataset_name, target_h_values, n_runs
            )

            # 分析结果
            analyze_and_visualize(results, dataset_name)

            all_results[dataset_name] = [asdict(r) for r in results]

        except Exception as e:
            print(f"  Error with {name}: {e}")
            import traceback
            traceback.print_exc()

    # 保存所有结果
    output_path = Path(__file__).parent / "semi_synthetic_hsweep_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'semi_synthetic_hsweep',
            'description': 'U-Shape validation on real features with edge rewiring',
            'target_h_values': target_h_values,
            'n_runs': n_runs,
            'results': all_results
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # 总结
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print("""
This experiment validates the U-Shape pattern on REAL NODE FEATURES.

If U-Shape is confirmed:
  -> Strong evidence that U-Shape is a real phenomenon, not just synthetic artifact
  -> Key result for TKDE submission

If U-Shape is NOT confirmed:
  -> Need to investigate why
  -> May indicate feature quality or dataset-specific effects
""")


if __name__ == '__main__':
    main()
