"""Part 2 - classification and evaluation (560 train / 240 test)."""
import numpy as np
from common import (get_split, train_logistic, predict_proba, bce, confusion, metrics, cost,
                    choose_k, knn_predict, write_results, write_csv)

S = get_split()
Ztr, Zte, ytr, yte = S["Ztr"], S["Zte"], S["ytr"], S["yte"]
R = []


def add(k, v):
    R.append((k, v))


# 2.1 split
add("train_rows", len(ytr)); add("train_positives", int(ytr.sum()))
add("test_rows", len(yte)); add("test_positives", int(yte.sum()))

# 2.2 logistic regression from scratch
w, b, L = train_logistic(Ztr, ytr, 0.1, 1000, record=(0, 1, 10, 100, 1000))
for e in (0, 1, 10, 100, 1000):
    add("loss_epoch_%d" % e, L[e])
for j in range(7):
    add("weight_X%d" % (j + 1), w[j])
add("bias", b)
ptest = predict_proba(Zte, w, b)
ptrain = predict_proba(Ztr, w, b)
add("test_bce", bce(yte, ptest))

# 2.3 learning-rate effect
for eta in (0.01, 0.1, 1.0):
    _, _, l = train_logistic(Ztr, ytr, eta, 200, record=(200,))
    add("loss_epoch_200_eta_%s" % eta, l[200])

# 2.4 odds ratios
for j in range(7):
    add("odds_ratio_X%d" % (j + 1), float(np.exp(w[j])))

# 2.5 two thresholds
for tau in (0.5, 0.3):
    tp, fp, tn, fn = confusion(yte, ptest >= tau)
    m = metrics(tp, fp, tn, fn)
    for k, v in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn), ("accuracy", m["accuracy"]),
                 ("precision", m["precision"]), ("recall", m["recall"]), ("f1", m["f1"]),
                 ("specificity", m["specificity"])):
        add("tau_%s_%s" % (tau, k), v)
add("blind_accuracy", float(np.mean(yte == 0)))

# 2.6 threshold sweep
rows = []
for t in range(1, 10):
    tau = t / 10
    tp, fp, tn, fn = confusion(yte, ptest >= tau - 1e-12)
    m = metrics(tp, fp, tn, fn)
    rows.append((round(tau, 1), tp, fp, tn, fn, m["recall"], m["fpr"]))
write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

# 2.7 KNN with 5-fold CV
table, best_k = choose_k(Ztr, ytr)
write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], table)
add("knn_best_k", best_k)
pred, _ = knn_predict(Ztr, ytr, Zte, best_k)
tp, fp, tn, fn = confusion(yte, pred)
m = metrics(tp, fp, tn, fn)
for k, v in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn), ("accuracy", m["accuracy"]),
             ("precision", m["precision"]), ("recall", m["recall"]), ("f1", m["f1"])):
    add("knn_%s" % k, v)

# 2.8 money-based threshold, chosen on TRAINING rows
best = None
for k in range(1, 20):
    tau = k / 20
    _, fp_t, _, fn_t = confusion(ytr, ptrain >= tau - 1e-12)
    c = cost(fp_t, fn_t)
    if best is None or c < best[1]:          # strict '<' keeps the smaller tau on ties
        best = (tau, c)
cost_tau, cost_train = best
add("cost_tau", cost_tau); add("cost_train_at_tau", cost_train)
_, fp1, _, fn1 = confusion(yte, ptest >= cost_tau - 1e-12)
add("cost_test_at_tau", cost(fp1, fn1))
_, fp2, _, fn2 = confusion(yte, ptest >= 0.5)
add("cost_test_at_0.5", cost(fp2, fn2))
add("cost_test_flag_nobody", cost(0, int(yte.sum())))

write_results("part2_results.txt", R)
