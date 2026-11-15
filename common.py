"""Shared helpers for the capstone. Importing this file never runs a Part."""
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def load_data():
    """Return (X, outcome, value): X is 800 x 7 (X1..X7), outcome and value are 1-D arrays."""
    with open(os.path.join(HERE, "data", "records.csv"), newline="") as f:
        rows = list(csv.reader(f))[1:]
    arr = np.array([[float(v) for v in r[1:]] for r in rows])
    return arr[:, 0:7], arr[:, 7].astype(int), arr[:, 8]


# ---------- output formatting ----------
def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "%.4f" % float(v)
    return str(v)


def write_results(name, items):
    """items: list of (key, value). One 'key = value' line per key."""
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="\n") as f:
        for k, v in items:
            f.write("%s = %s\n" % (k, fmt(v)))


def write_csv(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


# ---------- statistics ----------
def sample_sd(a, axis=0):
    return np.sqrt(np.sum((a - a.mean(axis=axis)) ** 2, axis=axis) / (a.shape[axis] - 1))


# ---------- Task 2.1: split + standardise ----------
def stratified_split(y):
    """Boolean mask of training rows: first floor(7*count/10) of each class, in file order."""
    train = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        n_tr = (7 * len(idx)) // 10
        train[idx[:n_tr]] = True
    return train


def get_split():
    X, y, v = load_data()
    tr = stratified_split(y)
    Xtr, Xte = X[tr], X[~tr]
    mu, sd = Xtr.mean(axis=0), sample_sd(Xtr)
    return dict(Xtr_raw=Xtr, Xte_raw=Xte, Ztr=(Xtr - mu) / sd, Zte=(Xte - mu) / sd,
                ytr=y[tr], yte=y[~tr], mu=mu, sd=sd, train_mask=tr)


# ---------- Task 2.2: logistic regression ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_logistic(Z, y, eta, updates, record=()):
    """Batch gradient descent from zero. Returns w, b and {epoch: loss} for epochs in record."""
    n, d = Z.shape
    w, b = np.zeros(d), 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(y, sigmoid(Z @ w + b))
    for ep in range(1, updates + 1):
        p = sigmoid(Z @ w + b)
        err = p - y
        w = w - eta * (Z.T @ err) / n
        b = b - eta * err.sum() / n
        if ep in record:
            losses[ep] = bce(y, sigmoid(Z @ w + b))
    return w, b, losses


def predict_proba(Z, w, b):
    return sigmoid(Z @ w + b)


# ---------- metrics ----------
def confusion(y, pred):
    y, pred = np.asarray(y), np.asarray(pred).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


def _div(a, b):
    return a / b if b else 0.0


def metrics(tp, fp, tn, fn):
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    return dict(accuracy=_div(tp + tn, tp + fp + tn + fn), precision=prec, recall=rec,
                f1=_div(2 * prec * rec, prec + rec), specificity=_div(tn, tn + fp),
                fpr=_div(fp, fp + tn))


def cost(fp, fn):
    return 50 * fp + 400 * fn


# ---------- Task 2.7: KNN ----------
def knn_neighbour_share(Zref, yref, Zq, K):
    """Share of positive neighbours among K nearest (ties by file order via stable sort)."""
    shares = np.empty(len(Zq))
    for i, q in enumerate(Zq):
        d = np.sqrt(np.sum((Zref - q) ** 2, axis=1))
        order = np.argsort(d, kind="stable")[:K]
        shares[i] = yref[order].sum() / K
    return shares


def knn_predict(Zref, yref, Zq, K):
    share = knn_neighbour_share(Zref, yref, Zq, K)
    pred = (share * K >= (K + 1) / 2).astype(int)
    return pred, share


def choose_k(Ztr, ytr):
    """5-fold CV on the training rows; fold = position mod 5. Returns [(K, mean_f1)], best K."""
    n = len(ytr)
    fold = np.arange(n) % 5
    table = []
    for K in range(1, 22, 2):
        f1s = []
        for f in range(5):
            ref, held = fold != f, fold == f
            pred, _ = knn_predict(Ztr[ref], ytr[ref], Ztr[held], K)
            f1s.append(metrics(*confusion(ytr[held], pred))["f1"])
        table.append((K, float(np.mean(f1s))))
    best = max(table, key=lambda t: (t[1], -t[0]))[0]
    return table, best
