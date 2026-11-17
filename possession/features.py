"""Per-frame features from tracking + Smart Tagging. All arrays are indexed by video frame."""

import numpy as np

FPS = 25


def rolling_nanmean(x, half):
    """Centred mean over +/- half frames, ignoring NaN. Cumulative-sum based, O(n)."""
    v = np.where(np.isnan(x), 0.0, x).astype(np.float64)
    c = (~np.isnan(x)).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0.0], np.cumsum(c)))
    n = len(x)
    i = np.arange(n)
    lo = np.maximum(0, i - half)
    hi = np.minimum(n, i + half + 1)
    s = cv[hi] - cv[lo]
    k = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / k
    out[k == 0] = np.nan
    return out


def ball_kinematics(bx, by):
    """Ball speed (m/s) from positions, lightly smoothed. NaN where the ball is not seen."""
    sx = rolling_nanmean(bx.astype(np.float64), 2)
    sy = rolling_nanmean(by.astype(np.float64), 2)
    sx[np.isnan(bx)] = np.nan
    sy[np.isnan(by)] = np.nan
    vx = np.gradient(sx) * FPS
    vy = np.gradient(sy) * FPS
    spd = np.sqrt(vx ** 2 + vy ** 2)
    # one-frame gaps make gradient explode; clip to physically possible
    spd = np.clip(spd, 0, 40)
    return spd


def nearest_by_team(px, py, bx, by, slot_team):
    """Distance (m) from the ball to the nearest home and nearest away player. NaN if unseen."""
    dx = px - bx[:, None]
    dy = py - by[:, None]
    d = np.sqrt(dx * dx + dy * dy)
    home = slot_team == 0
    with np.errstate(all="ignore"):
        dh = np.nanmin(np.where(home[None, :], d, np.nan), axis=1)
        da = np.nanmin(np.where(~home[None, :], d, np.nan), axis=1)
    return dh, da


def team_shape(px, py, ps, slot_team):
    """Collective features per team. Used by the self-supervised team model, and for motion."""
    home = slot_team == 0
    out = {}
    with np.errstate(all="ignore"):
        for name, m in (("h", home), ("a", ~home)):
            x = np.where(m[None, :], px, np.nan)
            y = np.where(m[None, :], py, np.nan)
            s = np.where(m[None, :], ps, np.nan)
            out[name + "_mx"] = np.nanmean(x, axis=1)
            out[name + "_my"] = np.nanmean(y, axis=1)
            out[name + "_sx"] = np.nanstd(x, axis=1)
            out[name + "_sy"] = np.nanstd(y, axis=1)
            out[name + "_spd"] = np.nanmean(s, axis=1)
            out[name + "_n"] = np.sum(~np.isnan(x), axis=1)
        allspd = np.nanmean(ps, axis=1)
    out["all_spd"] = allspd
    return out


def player_speed_from_positions(px, py):
    """Fallback if the raw file has no speed channel."""
    sx = np.gradient(px, axis=0) * FPS
    sy = np.gradient(py, axis=0) * FPS
    return np.clip(np.sqrt(sx ** 2 + sy ** 2), 0, 12).astype(np.float32)


def tag_features(events, n_frames, home_name, away_name, team_side):
    """Signed per-code indicators: +1 where a phase tagged with the home team covers the frame,
    -1 for away, 0 otherwise. The sign *meaning* (attacking vs defending team) is learnt later
    from pseudo-labels, not assumed."""
    codes = sorted({e["code"] for e in events if e["code"] != "SET PIECES"})
    F = np.zeros((n_frames, len(codes)), np.float32)
    idx = {c: i for i, c in enumerate(codes)}
    for e in events:
        if e["code"] not in idx:
            continue
        side = team_side(e["tags"].get("Team"), home_name, away_name)
        if side is None:
            continue
        s, t = max(0, e["start"]), min(n_frames, e["end"])
        F[s:t, idx[e["code"]]] += 1.0 if side == "H" else -1.0
    return np.clip(F, -1, 1), codes
