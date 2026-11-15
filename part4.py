"""Part 4 - model choice, boosting by hand and network sizing."""
import numpy as np
from common import (get_split, train_logistic, predict_proba, confusion, metrics,
                    choose_k, knn_predict, write_results, write_csv)

S = get_split()
Xtr, Xte = S["Xtr_raw"], S["Xte_raw"]
Ztr, Zte, ytr, yte = S["Ztr"], S["Zte"], S["ytr"], S["yte"]
R = []


def add(k, v):
    R.append((k, v))


# 4.1 which split is purest (training rows, original units)
def entropy(p_pos, n):
    if n == 0:
        return 0.0
    h = 0.0
    for p in (p_pos / n, 1 - p_pos / n):
        if p > 0:
            h -= p * np.log2(p)
    return h


def gini(p_pos, n):
    if n == 0:
        return 0.0
    p = p_pos / n
    return 1 - p ** 2 - (1 - p) ** 2


n_tr, pos_tr = len(ytr), int(ytr.sum())
root_h, root_g = entropy(pos_tr, n_tr), gini(pos_tr, n_tr)
add("root_entropy", root_h); add("root_gini", root_g)
cands = {"S1": Xtr[:, 5] == 1, "S2": Xtr[:, 6] == 1, "S3": Xtr[:, 2] > 0.5, "S4": Xtr[:, 0] <= 24}
rows, ig, gg = [], {}, {}
for name, yes in cands.items():
    ny, py = int(yes.sum()), int(ytr[yes].sum())
    nn, pn = int((~yes).sum()), int(ytr[~yes].sum())
    ig[name] = root_h - (ny / n_tr * entropy(py, ny) + nn / n_tr * entropy(pn, nn))
    gg[name] = root_g - (ny / n_tr * gini(py, ny) + nn / n_tr * gini(pn, nn))
    rows.append((name, ny, py, nn, pn, ig[name], gg[name]))
write_csv("part4_splits.csv", ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"], rows)
best_ig = max(ig, key=lambda k: (ig[k], 0))
best_gg = max(gg, key=lambda k: (gg[k], 0))
add("best_split_by_info_gain", best_ig); add("best_split_by_gini_gain", best_gg)

# 4.2 stump + voting on the 240 test rows
yes_tr = cands[best_ig]
rate_yes = float(ytr[yes_tr].mean()); rate_no = float(ytr[~yes_tr].mean())
add("stump_split", best_ig); add("stump_rate_yes", rate_yes); add("stump_rate_no", rate_no)
test_yes = {"S1": Xte[:, 5] == 1, "S2": Xte[:, 6] == 1, "S3": Xte[:, 2] > 0.5, "S4": Xte[:, 0] <= 24}[best_ig]
p_stump = np.where(test_yes, rate_yes, rate_no)

w, b, _ = train_logistic(Ztr, ytr, 0.1, 1000)
p_log = predict_proba(Zte, w, b)
_, best_k = choose_k(Ztr, ytr)
pred_knn, p_knn = knn_predict(Ztr, ytr, Zte, best_k)

pred_log = (p_log >= 0.3).astype(int)
pred_stump = (p_stump >= 0.3).astype(int)
hard = ((pred_log + pred_knn + pred_stump) >= 2).astype(int)
soft = (((p_log + p_knn + p_stump) / 3) >= 0.3 - 1e-12).astype(int)
vrows = []
for name, pr in (("logistic", pred_log), ("knn", pred_knn), ("stump", pred_stump),
                 ("hard_vote", hard), ("soft_vote", soft)):
    tp, fp, tn, fn = confusion(yte, pr)
    m = metrics(tp, fp, tn, fn)
    vrows.append((name, tp, fp, tn, fn, m["precision"], m["recall"], m["f1"]))
write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vrows)

# 4.3 two rounds of AdaBoost
yb = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])
w0 = np.full(8, 1 / 8)
ada_rows, alphas = [], []
w_before = w0
for rnd, h in ((1, h1), (2, h2)):
    eps = float(w_before[h != yb].sum())
    alpha = 0.5 * np.log((1 - eps) / eps)
    unnorm = w_before * np.exp(-alpha * yb * h)
    after = unnorm / unnorm.sum()
    for i in range(8):
        ada_rows.append((rnd, i + 1, float(w_before[i]), float(unnorm[i]), float(after[i])))
    add("round%d_error" % rnd, eps); add("round%d_alpha" % rnd, alpha)
    if rnd == 1:
        add("round1_misclassified_weight_after", float(after[h1 != yb].sum()))
    alphas.append(alpha)
    w_before = after
write_csv("part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], ada_rows)
ens = np.sign(alphas[0] * h1 + alphas[1] * h2)
add("ensemble_accuracy", float(np.mean(ens == yb)))

# 4.4 out-of-bag share
add("oob_fraction", (1 - 1 / 560) ** 560)

# 4.5 network sizing (counts only)
mlp = [("mlp_hidden1", 784 * 16 + 16), ("mlp_hidden2", 16 * 16 + 16), ("mlp_output", 16 * 10 + 10)]


def out_side(i, f, pad, stride):
    return (i - f + 2 * pad) // stride + 1


s1 = out_side(28, 5, 2, 1)           # 28
p1 = out_side(s1, 2, 0, 2)           # 14
s2 = out_side(p1, 5, 0, 1)           # 10
p2 = out_side(s2, 2, 0, 2)           # 5
flat = p2 * p2 * 16                  # 400
cnn = [("cnn_conv1", 5 * 5 * 1 * 6 + 6), ("cnn_conv2", 5 * 5 * 6 * 16 + 16),
       ("cnn_dense1", flat * 32 + 32), ("cnn_dense2", 32 * 10 + 10)]
write_csv("part4_architecture.csv", ["layer", "parameters"], mlp + cnn)
add("mlp_total", sum(p for _, p in mlp)); add("cnn_total", sum(p for _, p in cnn))
add("fc_layer_params", 128 * 128 * 3 * 1024 + 1024)
add("conv_layer_params", 3 * 3 * 3 * 64 + 64)
add("rnn_total", 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16)

write_results("part4_results.txt", R)
print("shapes:", s1, p1, s2, p2, flat)
