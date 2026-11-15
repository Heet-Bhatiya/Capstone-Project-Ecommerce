"""Part 3 - segmentation and dimensionality reduction (all 800 rows, X1..X5)."""
import numpy as np
from common import load_data, write_results, write_csv, sample_sd

X, y, value = load_data()
X5 = X[:, :5]
n = len(X5)
Z = (X5 - X5.mean(axis=0)) / sample_sd(X5)
R = []


def add(k, v):
    R.append((k, v))


# 3.1 K-means from scratch with pinned starting rows
def kmeans(Z, K):
    step = len(Z) // K
    cent = Z[[i * step for i in range(K)]].copy()
    labels = None
    passes = 0
    while True:
        passes += 1
        d = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1)             # argmin returns the lowest index on ties
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for k in range(K):
            members = Z[labels == k]
            if len(members):
                cent[k] = members.mean(axis=0)
    wcss = float(((Z - cent[labels]) ** 2).sum())
    return labels, cent, wcss, passes


wcss, fits = {}, {}
for K in range(2, 7):
    lab, cent, w, passes = kmeans(Z, K)
    wcss[K], fits[K] = w, (lab, passes)
    add("wcss_k%d" % K, w)

# 3.2 elbow
elbow_k = None
for K in (3, 4, 5):
    if wcss[K] - wcss[K + 1] < 0.5 * (wcss[K - 1] - wcss[K]):
        elbow_k = K
        break
add("elbow_k", elbow_k if elbow_k is not None else "none")

# 3.3 profile K = 3 segments, ordered by ascending mean X3
lab3 = fits[3][0]
order = sorted(range(3), key=lambda k: X5[lab3 == k, 2].mean())
seg_of = {k: "ABC"[i] for i, k in enumerate(order)}
seg_rows, seg_masks = [], {}
for k in order:
    m = lab3 == k
    name = seg_of[k]
    seg_masks[name] = m
    seg_rows.append((name, int(m.sum()), *[float(X5[m, j].mean()) for j in range(5)],
                     float(y[m].mean()), float(value[m].mean())))
    add("size_segment_%s" % name, int(m.sum()))
write_csv("part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"],
          seg_rows)

# 3.4 hierarchical clustering on the supplied table
names = ["E1", "E2", "E3", "E4", "E5", "E6"]
D = np.array([[0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
              [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
              [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
              [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
              [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
              [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def link(a, b, rule):
    d = [D[i, j] for i in a for j in b]
    return min(d) if rule == "single" else (max(d) if rule == "complete" else sum(d) / len(d))


def label(c):
    return "+".join(names[i] for i in sorted(c))


def agglomerate(rule):
    clusters = [(i,) for i in range(6)]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                h = link(clusters[i], clusters[j], rule)
                if best is None or h < best[0]:
                    best = (h, i, j)
        h, i, j = best
        a, b = clusters[i], clusters[j]
        first, second = (a, b) if min(a) < min(b) else (b, a)
        merges.append((first, second, h))
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [tuple(sorted(a + b))]
    return merges


hier_rows, cut_counts = [], {}
for rule in ("single", "complete", "average"):
    merges = agglomerate(rule)
    for no, (a, b, h) in enumerate(merges, 1):
        hier_rows.append((rule, no, label(a), label(b), h))
    heights = [m[2] for m in merges]
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    g = int(np.argmax(gaps))                 # gap between merge g+1 and g+2 -> g+1 merges below the cut
    cut_counts[rule] = 6 - (g + 1)
write_csv("part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hier_rows)
for rule in ("single", "complete", "average"):
    add("%s_clusters_at_cut" % rule, cut_counts[rule])

# 3.5 PCA by the seven steps
mean_vec = Z.mean(axis=0)                    # (1) all zeros after standardising
Xc = Z - mean_vec                            # (2) centre
S = Xc.T @ Xc / (n - 1)                      # (3) covariance
vals, vecs = np.linalg.eigh(S)               # (4) eigen-decomposition
idx = np.argsort(vals)[::-1]                 # (5) sort largest first
vals, vecs = vals[idx], vecs[:, idx]
for c in range(5):                           # sign convention
    if vecs[np.argmax(np.abs(vecs[:, c])), c] < 0:
        vecs[:, c] = -vecs[:, c]
W = vecs[:, :2]                              # (6) top two
P = Xc @ W                                   # (7) project
tot = float(vals.sum())
add("eigenvalue_sum", tot)
for i in range(5):
    add("eigenvalue_%d" % (i + 1), float(vals[i]))
ratio = vals / tot
for i in range(5):
    add("explained_ratio_%d" % (i + 1), float(ratio[i]))
cum = np.cumsum(ratio)
for i in range(5):
    add("cumulative_ratio_%d" % (i + 1), float(cum[i]))
add("components_for_80pct", int(np.argmax(cum >= 0.8)) + 1)
for c in (1, 2):
    for j in range(5):
        add("pc%d_loading_X%d" % (c, j + 1), float(vecs[j, c - 1]))
for r in range(3):
    add("row%d_pc1" % (r + 1), float(P[r, 0]))
    add("row%d_pc2" % (r + 1), float(P[r, 1]))
for name in "ABC":
    add("mean_pc1_segment_%s" % name, float(P[seg_masks[name], 0].mean()))

write_results("part3_results.txt", R)
print("kmeans passes:", {K: fits[K][1] for K in fits})
