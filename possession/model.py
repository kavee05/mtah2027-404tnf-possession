"""Self-supervised team model, unsupervised dead-ball motion model and the 3-state decoder."""

import numpy as np

STATES = ("H", "A", "D")


# --------------------------------------------------------------------------------------------
# Logistic regression (L2, Newton), numpy only
# --------------------------------------------------------------------------------------------

def fit_logistic(X, y, l2=1.0, iters=25):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = Xb.T @ (p - y) + reg * w
        W = p * (1 - p)
        Hm = (Xb * W[:, None]).T @ Xb + np.diag(reg + 1e-9)
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def predict_logodds(X, w):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def blocked_cv_accuracy(X, y, l2=1.0, k=5):
    """Contiguous-block CV, so neighbouring (correlated) frames never sit on both sides."""
    n = len(y)
    if n < 50:
        return float("nan")
    edges = np.linspace(0, n, k + 1).astype(int)
    correct = 0
    for i in range(k):
        te = np.zeros(n, bool)
        te[edges[i]:edges[i + 1]] = True
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        w = fit_logistic(X[~te], y[~te], l2)
        correct += np.sum((predict_logodds(X[te], w) > 0) == (y[te] > 0.5))
    return correct / n


# --------------------------------------------------------------------------------------------
# 1-D two-component Gaussian mixture (EM)
# --------------------------------------------------------------------------------------------

def gmm2(x, iters=200):
    x = x[np.isfinite(x)]
    mu = np.percentile(x, [20, 80]).astype(float)
    sd = np.array([x.std(), x.std()]) / 2 + 1e-3
    pi = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = (-0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(pi))
        m = ll.max(1, keepdims=True)
        r = np.exp(ll - m)
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9
        mu_new = (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu_new) ** 2).sum(0) / nk) + 1e-3
        pi = nk / nk.sum()
        if np.max(np.abs(mu_new - mu)) < 1e-7:
            mu = mu_new
            break
        mu = mu_new
    order = np.argsort(mu)
    return mu[order], sd[order], pi[order]


def gmm2_logit_low(x, params):
    """log P(low component | x) - log P(high component | x)."""
    mu, sd, pi = params
    l0 = -0.5 * ((x - mu[0]) / sd[0]) ** 2 - np.log(sd[0]) + np.log(pi[0])
    l1 = -0.5 * ((x - mu[1]) / sd[1]) ** 2 - np.log(sd[1]) + np.log(pi[1])
    return l0 - l1


# --------------------------------------------------------------------------------------------
# Viterbi over H / A / D
# --------------------------------------------------------------------------------------------

def viterbi(score, cost, start):
    """score: (T, 3) per-frame log-scores for H, A, D. cost: (3, 3) transition penalties (>=0,
    zero on the diagonal). start: (3,) initial log-scores. Returns int state per frame."""
    T = len(score)
    trans = -np.asarray(cost, float)
    delta = np.asarray(start, float) + score[0]
    back = np.empty((T, 3), np.int8)
    back[0] = np.arange(3)
    sc = score.astype(np.float64)
    for t in range(1, T):
        cand = delta[:, None] + trans          # from i (rows) to j (cols)
        b = cand.argmax(0)
        delta = cand[b, np.arange(3)] + sc[t]
        back[t] = b
    path = np.empty(T, np.int8)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
