"""Part 1 - estimation, inference and regression baseline (all 800 rows)."""
import numpy as np
from common import load_data, write_results, sample_sd

X, y, value = load_data()
n = len(value)
R = []  # results in order


def add(k, v):
    R.append((k, v))


# 1.1 estimate with an interval
mean_value = value.sum() / n
sd_value = float(sample_sd(value))
se_value = sd_value / np.sqrt(n)
add("n", n); add("mean_value", mean_value); add("sd_value", sd_value); add("se_value", se_value)
add("ci95_low", mean_value - 1.96 * se_value); add("ci95_high", mean_value + 1.96 * se_value)

# 1.2 test H0: mu = 8.0
z_stat = (mean_value - 8.0) / se_value
add("z_stat", z_stat); add("reject_h0", abs(z_stat) > 1.96)

# 1.3 sample size for margin of error 0.5
need = (1.96 * sd_value / 0.5) ** 2
n_required = int(np.ceil(need))
add("n_required", n_required)

# 1.4 closed-form simple regression of value on X4
x4 = X[:, 3]
mx, my = x4.mean(), value.mean()
cov = np.sum((x4 - mx) * (value - my)) / (n - 1)
var = np.sum((x4 - mx) ** 2) / (n - 1)
slope = cov / var
intercept = my - slope * mx
fitted = intercept + slope * x4
resid = value - fitted
tss = np.sum((value - my) ** 2)
ess = np.sum((fitted - my) ** 2)
rss = np.sum(resid ** 2)
add("reg_slope", slope); add("reg_intercept", intercept)
add("tss", tss); add("ess", ess); add("rss", rss)
add("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss)
add("r2", 1 - rss / tss); add("rmse", np.sqrt(rss / n)); add("mae", np.mean(np.abs(resid)))

# 1.5 gradient descent on standardised X4
sd4 = float(sample_sd(x4))
z = (x4 - mx) / sd4
m, c = 0.0, 0.0


def J(m, c):
    return float(np.mean((value - (m * z + c)) ** 2))


add("gd_J0", J(m, c))
snap = {}
for t in range(1, 501):
    e = value - (m * z + c)
    gm = -2.0 / n * np.sum(z * e)
    gc = -2.0 / n * np.sum(e)
    m, c = m - 0.1 * gm, c - 0.1 * gc
    if t in (1, 10, 100, 500):
        snap[t] = J(m, c)
for t in (1, 10, 100, 500):
    add("gd_J%d" % t, snap[t])
gd_slope = m / sd4
gd_intercept = c - m * mx / sd4
add("gd_slope", gd_slope); add("gd_intercept", gd_intercept)
add("gd_slope_gap", abs(gd_slope - slope))


# 1.6 funnel check and log remedy
def halves_sd(fit, res):
    order = np.argsort(fit, kind="stable")
    lo, hi = res[order[:400]], res[order[400:]]
    return float(sample_sd(lo)), float(sample_sd(hi))


lo, hi = halves_sd(fitted, resid)
add("resid_sd_lower_raw", lo); add("resid_sd_upper_raw", hi); add("resid_sd_ratio_raw", hi / lo)

lx, ly = np.log(x4), np.log(value)
lmx, lmy = lx.mean(), ly.mean()
lslope = (np.sum((lx - lmx) * (ly - lmy)) / (n - 1)) / (np.sum((lx - lmx) ** 2) / (n - 1))
lint = lmy - lslope * lmx
lfit = lint + lslope * lx
lres = ly - lfit
ltss, lrss = np.sum((ly - lmy) ** 2), np.sum(lres ** 2)
loglog_r2 = 1 - lrss / ltss
add("loglog_intercept", lint); add("loglog_slope", lslope)
add("loglog_r2", loglog_r2); add("loglog_rmse", np.sqrt(lrss / n))
llo, lhi = halves_sd(lfit, lres)
add("resid_sd_lower_log", llo); add("resid_sd_upper_log", lhi); add("resid_sd_ratio_log", lhi / llo)

# 1.7 multiple regression (normal equation) on ln(value)
design = np.column_stack([np.ones(n), np.log(X[:, 3]), X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(design.T @ design, design.T @ ly)
for i, b in enumerate(beta):
    add("mr_beta_%d" % i, b)
mres = ly - design @ beta
mr_r2 = 1 - np.sum(mres ** 2) / ltss
p = 7
add("mr_r2", mr_r2)
add("mr_adj_r2", 1 - (1 - mr_r2) * (n - 1) / (n - p - 1))
add("loglog_adj_r2", 1 - (1 - loglog_r2) * (n - 1) / (n - 1 - 1))

write_results("part1_results.txt", R)
