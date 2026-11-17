"""Out-of-sample check of our pipeline on Metrica's public sample games (github.com/metrica-sports/sample-data).

Ground truth is built mechanically from each game's human event log (allowed: other games):
  * possession = team of the latest PASS / CARRY / SET PIECE / SHOT / RECOVERY
  * dead ball  = from BALL OUT, FAULT RECEIVED or a goal until the next SET PIECE (if that
                 restart comes within 60 s; otherwise play went on, e.g. advantage)
  * graded     = from each period's kick-off to its last event
The pipeline is run twice per game: without any tagging, and with a synthetic Smart-Tagging
stand-in (a 5 s SET PIECES window around each restart, team = taker, N/A for throw-ins), since
these games ship no Smart Tagging.

    git clone --depth 1 https://github.com/metrica-sports/sample-data.git ../metrica-sample-data
    python validation/metrica_sample_games.py --data ../metrica-sample-data/data
"""
import argparse
import json
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evaluation"))
from possession import data as D  # noqa: E402
from possession import pipeline as PL  # noqa: E402
import possession_metrics as PM  # noqa: E402

PL.log = lambda *a, **k: None
ROOT = None
PITCH = (105.0, 68.0)
FPS = 25
ANCHORS = {"PASS", "CARRY", "SET PIECE", "SHOT", "RECOVERY"}


# ------------------------------------------------------------------ loaders
def load_csv_game(g):
    base = "{}/Sample_Game_{}/Sample_Game_{}_".format(ROOT, g, g)
    cols, xs, ys, teams = None, [], [], []
    frames = None
    for side, t in (("Home", 0), ("Away", 1)):
        df = pd.read_csv(base + "RawTrackingData_{}_Team.csv".format(side), skiprows=2)
        frames = df["Frame"].to_numpy()
        names = list(df.columns)
        for i, c in enumerate(names):
            if c.startswith("Player"):
                xs.append(df.iloc[:, i].to_numpy(float))
                ys.append(df.iloc[:, i + 1].to_numpy(float))
                teams.append(t)
        if side == "Home":
            bi = names.index("Ball")
            bxn = df.iloc[:, bi].to_numpy(float)
            byn = df.iloc[:, bi + 1].to_numpy(float)
    n = int(frames.max())
    rows = frames - 1
    px = np.full((n, len(xs)), np.nan, np.float32)
    py = np.full((n, len(xs)), np.nan, np.float32)
    px[rows] = np.column_stack(xs) * PITCH[0]
    py[rows] = np.column_stack(ys) * PITCH[1]
    bx = np.full(n, np.nan, np.float32)
    by = np.full(n, np.nan, np.float32)
    bx[rows] = bxn * PITCH[0]
    by[rows] = byn * PITCH[1]
    ev = pd.read_csv(base + "RawEventsData.csv")
    events = [{"team": 0 if r.Team == "Home" else 1, "type": str(r.Type).upper(),
               "sub": str(r.Subtype).upper() if isinstance(r.Subtype, str) else "",
               "period": int(r.Period), "start": int(r._5) - 1, "end": int(r._7) - 1}
              for r in ev.itertuples()]
    return dict(px=px, py=py, bx=bx, by=by, slot_team=np.array(teams, np.int8), n=n,
                events=events)


def load_epts_game():
    base = "{}/Sample_Game_3/Sample_Game_3_".format(ROOT)
    meta = D.read_meta(base + "metadata.xml")
    with open(base + "tracking.txt") as fh:
        n = sum(1 for _ in fh)
    raw = D.read_raw(base + "tracking.txt", len(meta["slot_team"]), n, PITCH)
    team_ids = [t[0] for t in meta["teams"]]
    js = json.load(open(base + "events.json"))["data"]
    events = []
    for e in js:
        if not e.get("team"):
            continue
        subs = e.get("subtypes")
        if isinstance(subs, dict):
            sub = subs.get("name", "")
        elif isinstance(subs, list):
            sub = "-".join(s.get("name", "") for s in subs)
        else:
            sub = ""
        events.append({"team": 0 if e["team"]["id"] == team_ids[0] else 1,
                       "type": e["type"]["name"].upper(), "sub": sub.upper(),
                       "period": int(e["period"]), "start": int(e["start"]["frame"]) - 1,
                       "end": int(e["end"]["frame"]) - 1})
    return dict(px=raw["px"], py=raw["py"], bx=raw["bx"], by=raw["by"],
                slot_team=meta["slot_team"], n=n, events=events)


# ------------------------------------------------------------------ ground truth
def build_gt(G):
    n, ev = G["n"], sorted(G["events"], key=lambda e: e["start"])
    gt = np.full(n, "X", dtype="<U1")
    blocks, kick = [], {}
    for p in sorted({e["period"] for e in ev}):
        pe = [e for e in ev if e["period"] == p]
        ko = next(e["start"] for e in pe if e["type"] == "SET PIECE")
        last = max(max(e["end"], e["start"]) for e in pe)
        blocks.append((p, ko, min(last, n - 1)))
        kick[p] = ko
        anchors = [e for e in pe if e["type"] in ANCHORS and e["start"] >= ko]
        for i, e in enumerate(anchors):
            nxt = anchors[i + 1]["start"] if i + 1 < len(anchors) else last + 1
            gt[e["start"]:nxt] = "H" if e["team"] == 0 else "A"
        sps = [e["start"] for e in pe if e["type"] == "SET PIECE"]
        for e in pe:
            goal = e["type"] == "SHOT" and "GOAL" in e["sub"] and "OFF" not in e["sub"]
            if e["type"] in ("BALL OUT", "FAULT RECEIVED") or goal:
                s = e["end"] if goal else e["start"]
                restart = [f for f in sps if f > s]
                if restart and restart[0] - s <= 60 * FPS:
                    gt[s:restart[0]] = "D"
    return gt, blocks, kick


def synthetic_tags(G):
    out = []
    for e in G["events"]:
        if e["type"] != "SET PIECE":
            continue
        team = "N/A" if "THROW" in e["sub"] else ("Home" if e["team"] == 0 else "Away")
        out.append({"code": "SET PIECES", "start": e["start"] - 62, "end": e["start"] + 63,
                    "tags": {"Team": team}})
    return out


# ------------------------------------------------------------------ run
def score(gt, pred):
    r = PM.evaluate(PM.label_codes(gt), PM.label_codes(pred), bootstrap_n=200)
    return r


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="sample-data/data folder")
    ROOT = ap.parse_args().data
    games = {"Game 1": load_csv_game(1), "Game 2": load_csv_game(2), "Game 3": load_epts_game()}
    rows = []
    for name, G in games.items():
        gt, blocks, kick = build_gt(G)
        ps = PL.F.player_speed_from_positions(G["px"], G["py"])
        nposs = PM.possession_stats(PM.label_codes(gt), PM.valid_runs(PM.label_codes(gt)))
        print("{}: {} frames, ball seen {:.0f}%, GT {}".format(
            name, G["n"], 100 * np.mean(~np.isnan(G["bx"])), nposs), flush=True)
        for cond, tags in (("no tags", []), ("set-piece tags", synthetic_tags(G))):
            pred = PL.decode(G["px"], G["py"], ps, G["bx"], G["by"], G["slot_team"], tags,
                             "Home", "Away", G["n"], PITCH, blocks, kick)
            r = score(gt, pred)
            rows.append(dict(game=name, tags=cond,
                             spellF1=round(r["spell_f1_mean"], 3),
                             macroF1=round(r["macro_f1"], 3),
                             team_acc=round(r["team_acc_gt_alive"], 3),
                             alive_dead_F1=round(r["alive_dead_f1"], 3),
                             poss=r["n_pred_spells"], gt_poss=r["n_gt_spells"]))
            print("  ", rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REPO, "validation", "results_metrica_sample_games.csv"), index=False)
    print()
    print(df.groupby("tags")[["spellF1", "macroF1", "team_acc", "alive_dead_F1"]]
          .mean().round(3))


if __name__ == "__main__":
    main()
