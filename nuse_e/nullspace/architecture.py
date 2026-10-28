"""Quy tắc suy kiến trúc từ phổ PCA: exit theo mức năng lượng (không cắt giữa cặp D4), block theo năng lượng phi
tuyến, nhánh tuyến tính theo R².
"""
import math

def allocate_blocks(energy, linear_r2, groups, total_blocks):
    if total_blocks < len(groups):
        raise ValueError("Need at least one block per budget")
    start, nl = 0, []
    for size in groups:
        nl.append(sum(energy[i] * (1 - max(0.0, linear_r2[i])) for i in range(start, start + size)))
        start += size
    share = [total_blocks * e / sum(nl) for e in nl]
    blocks = [max(1, math.floor(s)) for s in share]
    while sum(blocks) < total_blocks:
        rem = [s - b for s, b in zip(share, blocks)]
        blocks[rem.index(max(rem))] += 1
    while sum(blocks) > total_blocks:
        over = [b - s if b > 1 else -1e9 for s, b in zip(share, blocks)]
        blocks[over.index(max(over))] -= 1
    return blocks, nl


def admissible_cuts(eigenvalues, rel_tol=1e-6):
    d = len(eigenvalues)
    cuts = [k for k in range(1, d) if eigenvalues[k - 1] - eigenvalues[k] > rel_tol * eigenvalues[k - 1]]
    return cuts + [d]


def choose_budgets(eigenvalues, energy_targets=(0.5, 0.8, 0.9, 1.0), rel_tol=1e-6):
    total = sum(eigenvalues)
    cuts = admissible_cuts(eigenvalues, rel_tol)
    budgets = []
    for target in energy_targets:
        k = next(k for k in cuts if sum(eigenvalues[:k]) / total >= target - 1e-12)
        if k not in budgets:
            budgets.append(k)
    return budgets


def derive_architecture(eigenvalues, linear_r2, *, energy_targets=(0.5, 0.8, 0.9, 1.0), total_blocks=7,
                        r2_min=0.05, rel_tol=1e-6, min_share=0.5):
    budgets = choose_budgets(eigenvalues, energy_targets, rel_tol)

    def nonlinear_energy(lo, hi):
        return sum(eigenvalues[i] * (1 - max(0.0, linear_r2[i])) for i in range(lo, hi))

    while len(budgets) > 1:
        edges = [0] + budgets
        e = [nonlinear_energy(edges[g], edges[g + 1]) for g in range(len(budgets))]
        share = [total_blocks * v / sum(e) for v in e]
        g = min(range(len(budgets) - 1), key=lambda i: share[i])
        if share[g] >= min_share and len(budgets) <= total_blocks:
            break
        budgets.pop(g)
    groups = [b - a for a, b in zip([0] + budgets[:-1], budgets)]
    blocks, nonlinear = allocate_blocks(eigenvalues, linear_r2, groups, total_blocks)
    linear, group_r2, start = [], [], 0
    for size in groups:
        lam = eigenvalues[start:start + size]
        r2 = sum(l * max(0.0, r) for l, r in zip(lam, linear_r2[start:start + size])) / sum(lam)
        group_r2.append(r2)
        linear.append(r2 >= r2_min)
        start += size
    return {"budgets": budgets, "groups": groups, "blocks": blocks, "linear": linear,
            "group_linear_r2": group_r2, "nonlinear_energy": nonlinear,
            "rule": {"energy_targets": list(energy_targets), "total_blocks": total_blocks,
                     "r2_min": r2_min, "rel_tol": rel_tol, "min_share": min_share}}


def architecture_from_stats(stats, width=32, block="plain", alpha="uniform", **rule):
    """Rule output plus the trunk hyper-parameters (width, block kind, loss weights) that the rule does not set."""
    lam = stats["eigenvalues"].mean(0).tolist()
    r2 = stats["r2"].mean(0).tolist()
    arch = derive_architecture(lam, r2, **rule)
    arch["method"] = stats["method"]
    arch["retained_energy"] = [sum(lam[:k]) / sum(lam) for k in arch["budgets"]]
    arch["width"], arch["block"], arch["alpha"] = width, block, alpha
    return arch


def loss_alpha(spec, n_budgets):
    """Thm. 4 holds for any alpha >= 0: 'uniform' = every budget counts once; 'top' = image MSE at K max only."""
    if spec in (None, "uniform"):
        return [1.0] * n_budgets
    if spec == "top":
        return [0.0] * (n_budgets - 1) + [1.0]
    return list(spec)
