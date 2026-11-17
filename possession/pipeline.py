"""Ball possession for every frame: tracking + Smart Tagging -> frame,label CSV.

    python -m possession.pipeline --out submissions/TEAM.csv

See TECHNICAL.md for the reasoning behind every component and constant.
"""

import argparse
import json
import os
import sys
import time

import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from possession import data as D  # noqa: E402
from possession import features as F  # noqa: E402
from possession import model as M  # noqa: E402

FPS = 25
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------------------------------
# Constants. Chosen from football / tracking priors before looking at any labels -- never
# fitted to the public ground truth (RULES.md). Units: metres, m/s, nats.
# ------------------------------------------------------------------------------------------
P = dict(
    # proximity evidence
    sigma_ctrl=1.25,       # m: a ball within ~1 m of a tracked body centre is "at the feet"
    w_prox=1.0,            # nats per frame at full control
    fast_ball=8.0,         # m/s: above this a nearby player is more likely being passed by
    # self-supervised team model (shape + tagging)
    w_team=1.0 / FPS,      # ~1 effective observation per second -- frames are correlated
    smooth_shape=FPS // 2,
    # pseudo-label definition for the team model
    pl_close=1.0, pl_far=3.0, pl_maxspeed=6.0,
    # dead ball
    out_margin_hard=1.0, out_margin_soft=0.3,   # m outside the lines
    w_out_hard=2.0, w_out_soft=0.5,
    w_motion=0.06,         # nats/frame per unit of motion log-odds
    motion_smooth=FPS,
    w_ctrl_live=0.3,       # a moving ball at someone's feet argues against dead ball
    # set pieces from Smart Tagging
    w_sp_dead=0.25, w_sp_live=0.5, w_sp_team=0.3,
    sp_live_frames=2 * FPS, sp_team_frames=int(1.6 * FPS),
    kick_speed=4.0, still_speed=1.5, still_frames=10,
    # decoder transition penalties (nats)
    c_switch=8.0, c_to_dead=8.0, c_from_dead=4.0,
)

# Kickoff video frames, published in the challenge repo (DATA.md / FAQ.md). Frames of a graded
# block before its kickoff are pre-match: the ball is not in play.
KICKOFF = {1: 11185, 2: 113742}


def log(msg):
    print(msg, flush=True)


def build(data_dir=None, game=GAME, diag=None):
    t0 = time.time()
    diag = {} if diag is None else diag
    n = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    L, W = pitch

    meta_path, raw_path, st_path = require(["atd_meta", "atd_raw", "smart_tagging"], game,
                                           data_dir)
    meta = D.read_meta(meta_path)
    log("teams: H = {} | A = {}   ({} slots, mapping from {})".format(
        meta["home_name"], meta["away_name"], len(meta["slot_team"]),
        meta["slot_mapping_source"]))
    log("slots per team: home {}  away {}".format(
        int((meta["slot_team"] == 0).sum()), int((meta["slot_team"] == 1).sum())))

    raw = D.read_raw(raw_path, len(meta["slot_team"]), n, pitch, meta["speed_unit"])
    log("raw: {} records, frame_id {}..{}, {} fields/player, {} fields/ball  [{:.0f}s]".format(
        raw["n_records"], *raw["frame_id_range"], raw["player_fields"], raw["ball_fields"],
        time.time() - t0))
    px, py, ps, bx, by = raw["px"], raw["py"], raw["ps"], raw["bx"], raw["by"]
    if raw["player_fields"] < 3 or np.all(np.isnan(ps)):
        ps = F.player_speed_from_positions(px, py)
    slot_team = meta["slot_team"]

    events = D.read_smart_tagging(st_path, FPS)
    side = lambda t: D.team_side(t, meta["home_name"], meta["away_name"])  # noqa: E731
    sides = {}
    for e in events:
        v = e["tags"].get("Team")
        sides[v] = side(v)
    log("smart tagging: {} instances, Team -> side {}".format(len(events), sides))

    # ---------------------------------------------------------------- graded mask
    blocks = half_video_blocks(game)
    graded = np.zeros(n, bool)
    half_of = np.zeros(n, np.int8)
    for h, a, b in blocks:
        graded[a:b + 1] = True
        half_of[a:b + 1] = h

    # ---------------------------------------------------------------- base features
    bvis = ~np.isnan(bx)
    bspd = F.ball_kinematics(bx, by)
    dh, da = F.nearest_by_team(px, py, bx, by, slot_team)
    sh = F.team_shape(px, py, ps, slot_team)
    log("ball visible on {:.1f}% of graded frames; out-of-[0,1] ball coords on {:.2f}%".format(
        100 * bvis[graded].mean(),
        100 * np.mean(((bx < 0) | (bx > L) | (by < 0) | (by > W))[graded & bvis])))
    log("ball min/max x {:.1f}/{:.1f}  y {:.1f}/{:.1f} m".format(
        np.nanmin(bx), np.nanmax(bx), np.nanmin(by), np.nanmax(by)))

    # ---------------------------------------------------------------- 1. proximity evidence
    s2 = 2 * P["sigma_ctrl"] ** 2
    eh = np.where(np.isnan(dh), 0.0, np.exp(-(np.nan_to_num(dh, nan=99) ** 2) / s2))
    ea = np.where(np.isnan(da), 0.0, np.exp(-(np.nan_to_num(da, nan=99) ** 2) / s2))
    bs = np.nan_to_num(bspd, nan=0.0)
    g = np.where(bs <= P["fast_ball"], 1.0,
                 np.clip(1.0 - (bs - P["fast_ball"]) / 12.0, 0.2, 1.0))
    llr_prox = P["w_prox"] * (eh - ea) * g

    # ---------------------------------------------------------------- 2. self-supervised team
    k = P["smooth_shape"]
    rm = lambda x: F.rolling_nanmean(np.asarray(x, np.float64), k)  # noqa: E731
    shape_cols = [
        rm(sh["h_mx"] - sh["a_mx"]),
        rm(sh["h_sx"] - sh["a_sx"]),
        rm(sh["h_sy"] - sh["a_sy"]),
        rm(sh["h_spd"] - sh["a_spd"]),
        rm((sh["h_mx"] + sh["a_mx"]) / 2 - L / 2),
    ]
    TF, codes = F.tag_features(events, n, meta["home_name"], meta["away_name"], D.team_side)
    X = np.column_stack(shape_cols + [TF[:, i] for i in range(TF.shape[1])])
    X = np.nan_to_num(X, nan=0.0)
    names = ["dmx", "dsx", "dsy", "dspd", "tilt"] + ["tag:" + c for c in codes]

    close = np.minimum(np.nan_to_num(dh, nan=99), np.nan_to_num(da, nan=99))
    far = np.maximum(np.nan_to_num(dh, nan=99), np.nan_to_num(da, nan=99))
    pl_mask = (graded & bvis & (close < P["pl_close"]) & (far > P["pl_far"])
               & (bs < P["pl_maxspeed"]))
    pl_y = (np.nan_to_num(dh, nan=99) < np.nan_to_num(da, nan=99)).astype(float)
    log("pseudo-labels: {} confident frames ({:.1f}% H)".format(
        int(pl_mask.sum()), 100 * pl_y[pl_mask].mean() if pl_mask.any() else float("nan")))

    llr_team = np.zeros(n)
    diag["team_model"] = {}
    for h, a, b in blocks:
        idx = np.arange(a, b + 1)
        m = pl_mask[idx]
        Xh = X[idx]
        mu, sd = Xh.mean(0), Xh.std(0) + 1e-6
        Z = (Xh - mu) / sd
        w = M.fit_logistic(Z[m], pl_y[idx][m], l2=5.0)
        acc = M.blocked_cv_accuracy(Z[m], pl_y[idx][m], l2=5.0)
        lo = np.clip(M.predict_logodds(Z, w), -4, 4)
        llr_team[idx] = P["w_team"] * lo
        coefs = {nm: round(float(c), 3) for nm, c in zip(names, w[:-1])}
        diag["team_model"][h] = {"cv_acc": round(float(acc), 3), "coef": coefs,
                                 "n_train": int(m.sum())}
        log("  half {} team model: blocked-CV acc on pseudo-labels {:.3f} (n={})".format(
            h, acc, int(m.sum())))
        top = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:6]
        log("    top coefs: {}".format(top))

    # ---------------------------------------------------------------- 3. dead-ball evidence
    llr_dead = np.zeros(n)
    outd = np.maximum.reduce([-bx, bx - L, -by, by - W])  # metres outside the pitch, <0 inside
    outd = np.where(bvis, outd, -99)
    llr_dead += np.where(outd > P["out_margin_hard"], P["w_out_hard"],
                         np.where(outd > P["out_margin_soft"], P["w_out_soft"], 0.0))

    mot = np.log(np.clip(F.rolling_nanmean(np.asarray(sh["all_spd"], np.float64),
                                           P["motion_smooth"]), 0.05, None))
    gp = M.gmm2(mot[graded])
    mlo = np.clip(M.gmm2_logit_low(mot, gp), -4, 4)
    mlo = np.nan_to_num(mlo, nan=0.0)
    moving_ball = bvis & (bs > P["still_speed"])
    llr_dead += P["w_motion"] * mlo * np.where(moving_ball, 0.3, 1.0)
    diag["motion_gmm"] = {"mean_speed_ms": [round(float(np.exp(v)), 2) for v in gp[0]],
                          "weight": [round(float(v), 3) for v in gp[2]]}
    log("motion GMM: low/high mean player speed {} m/s, weights {}".format(
        diag["motion_gmm"]["mean_speed_ms"], diag["motion_gmm"]["weight"]))

    llr_dead -= P["w_ctrl_live"] * np.maximum(eh, ea) * moving_ball

    # Set pieces: dead before the kick, live after, restarting team owns it.
    n_sp, n_kick = 0, 0
    for e in events:
        if e["code"] != "SET PIECES":
            continue
        s, t = max(0, e["start"]), min(n, e["end"])
        if t <= s or not graded[s:t].any():
            continue
        n_sp += 1
        kick = None
        still = 0
        for f in range(s, min(n, t + FPS)):
            if bvis[f] and bs[f] < P["still_speed"]:
                still += 1
            elif bvis[f] and bs[f] > P["kick_speed"] and still >= P["still_frames"]:
                kick = f
                break
        if kick is None:
            kick = (s + t) // 2
        else:
            n_kick += 1
        llr_dead[s:kick] += P["w_sp_dead"]
        llr_dead[kick:kick + P["sp_live_frames"]] -= P["w_sp_live"]
        team = side(e["tags"].get("Team"))
        if team is None:
            w_ = slice(kick, kick + 10)
            mh, ma = np.nanmean(dh[w_]) if np.any(~np.isnan(dh[w_])) else np.nan, \
                np.nanmean(da[w_]) if np.any(~np.isnan(da[w_])) else np.nan
            if np.isfinite(mh) and np.isfinite(ma):
                team = "H" if mh < ma else "A"
        if team is not None:
            sgn = 1.0 if team == "H" else -1.0
            llr_team[kick:kick + P["sp_team_frames"]] += sgn * P["w_sp_team"]
    log("set pieces in play: {}, kick located from ball motion in {}".format(n_sp, n_kick))

    # ---------------------------------------------------------------- 4. decode
    llr_ha = llr_prox + llr_team
    score = np.column_stack([0.5 * llr_ha, -0.5 * llr_ha, llr_dead])
    c = P
    cost = np.array([[0, c["c_switch"], c["c_to_dead"]],
                     [c["c_switch"], 0, c["c_to_dead"]],
                     [c["c_from_dead"], c["c_from_dead"], 0]])
    labels = np.full(n, "X", dtype="<U1")
    for h, a, b in blocks:
        ko = max(a, KICKOFF.get(h, a))
        if ko > a:
            labels[a:ko] = "D"
        sl = slice(ko, b + 1)
        path = M.viterbi(score[sl], cost, start=np.array([-5.0, -5.0, 0.0]))
        labels[sl] = np.array(M.STATES)[path]
    log("decoded  [{:.0f}s total]".format(time.time() - t0))
    return labels, diag


def summarise(labels, game=GAME):
    out = {}
    for h, a, b in half_video_blocks(game):
        seg = labels[a:b + 1]
        ch = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], ch))
        ends = np.concatenate((ch, [len(seg)]))
        runs = [(seg[s], e - s) for s, e in zip(starts, ends)]
        poss = [r for r in runs if r[0] in "HA"]
        dead = [r for r in runs if r[0] == "D"]
        out[h] = {
            "possessions": len(poss),
            "mean_poss_s": float(round(np.mean([r[1] for r in poss]) / FPS, 1)) if poss else 0.0,
            "dead_segments": len(dead),
            "H%": float(round(100 * np.mean(seg == "H"), 1)),
            "A%": float(round(100 * np.mean(seg == "A"), 1)),
            "D%": float(round(100 * np.mean(seg == "D"), 1)),
        }
    return out


def write_csv(path, labels):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None, help="defaults to $MTAH_DATA_DIR or ./data")
    ap.add_argument("--out", default=os.path.join(ROOT, "submissions", "submission.csv"))
    args = ap.parse_args()
    diag = {}
    labels, diag = build(args.data_dir, diag=diag)
    write_csv(args.out, labels)
    summ = summarise(labels)
    diag["summary"] = summ
    diag["params"] = P
    with open(os.path.splitext(args.out)[0] + "_diagnostics.json", "w") as fh:
        json.dump(diag, fh, indent=2, default=str)
    for h, s in summ.items():
        log("half {}: {}".format(h, s))
    log("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
