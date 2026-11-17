"""Readers for the provided match assets. numpy + pandas + stdlib only.

Everything returned here is indexed by VIDEO frame (0 .. n-1), i.e. by submission row.
ATD frame_id counts from 1, so video_frame = frame_id - 1 (see challenge DATA.md).
"""

import io
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------------
# ATD metadata
# --------------------------------------------------------------------------------------------

def _strip_ns(root):
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def read_meta(meta_path):
    """Teams, per-slot team (0 = home, 1 = away), and the channel layout of the raw file.

    The slot -> player mapping is read from the DataFormatSpecification + PlayerChannels when the
    file has them, which is exactly what the raw file is written from. If that fails it falls
    back to the order of <Player> elements, which is what the official baseline assumes.
    """
    root = _strip_ns(ET.parse(meta_path).getroot())

    teams = [(t.get("id"), (t.findtext("Name") or "").strip()) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(teams)))
    home_id = teams[0][0]  # FIFA/EPTS convention, also what the baseline and kloppy use

    players = [(p.get("id"), p.get("teamId")) for p in root.iter("Player")]
    team_of_player = {pid: (0 if tid == home_id else 1) for pid, tid in players}

    # PlayerChannel id -> player id
    chan_to_player = {}
    for pc in root.iter("PlayerChannel"):
        chan_to_player[pc.get("id")] = pc.get("playerId")

    # Raw layout: one group per player, in DataFormatSpecification order.
    groups, ball_fields = [], []
    spec = next(root.iter("DataFormatSpecification"), None)
    if spec is not None:
        for reg in spec.iter("SplitRegister"):
            refs = [c.get("playerChannelId") for c in reg.findall("PlayerChannelRef")]
            if refs:
                groups.append(refs)
        ball_fields = [c.get("channelId") for c in spec.iter("BallChannelRef")]

    slot_team, slot_fields, how = None, None, "player order (baseline convention)"
    if groups and chan_to_player:
        try:
            slot_team = np.array([team_of_player[chan_to_player[g[0]]] for g in groups],
                                 dtype=np.int8)
            slot_fields = [[re.sub(r"^.*?_", "", ref) if ref else "" for ref in g] for g in groups]
            how = "DataFormatSpecification + PlayerChannels"
        except KeyError:
            slot_team = None
    if slot_team is None:
        slot_team = np.array([team_of_player[pid] for pid, _ in players], dtype=np.int8)

    speed_unit = "m/s"
    for ch in root.iter("Channel"):
        unit = (ch.findtext("Unit") or "").lower()
        name = (ch.findtext("Name") or "").lower()
        if "speed" in name or ch.get("id") in ("s", "speed"):
            if "kph" in unit or "km" in unit:
                speed_unit = "kph"

    pitch = None
    for tag in ("FieldSize", "PitchSize"):
        el = next(root.iter(tag), None)
        if el is not None:
            try:
                pitch = (float(el.findtext("Width")), float(el.findtext("Height")))
            except (TypeError, ValueError):
                pass

    return {
        "teams": teams,
        "home_name": teams[0][1],
        "away_name": teams[1][1],
        "slot_team": slot_team,
        "slot_fields": slot_fields,
        "ball_fields": ball_fields,
        "slot_mapping_source": how,
        "speed_unit": speed_unit,
        "meta_pitch": pitch,
    }


# --------------------------------------------------------------------------------------------
# ATD raw
# --------------------------------------------------------------------------------------------

def read_raw(raw_path, n_slots, n_frames, pitch, speed_unit="m/s"):
    """Positions in metres, speeds in m/s, ball in metres. NaN where not observed.

    Returns dict with:
      px, py  : (n_frames, n_slots) float32 metres
      ps      : (n_frames, n_slots) float32 m/s (NaN if the file has no speed channel)
      bx, by  : (n_frames,) float32 metres
    """
    length, width = pitch
    with open(raw_path, "r") as fh:
        text = fh.read()

    lines = text.splitlines()
    # Discover group widths from the first line with data
    sample = next(l for l in lines if l.count(":") == 2)
    _, pl, bl = sample.split(":")
    pw = len(pl.split(";")[0].split(","))
    bw = len(bl.split(","))
    n_groups = len(pl.split(";"))
    if n_groups != n_slots:
        raise ValueError("raw file has {} player groups, metadata has {} slots".format(
            n_groups, n_slots))

    flat = text.replace(":", ",").replace(";", ",")
    ncol = 1 + n_slots * pw + bw
    df = pd.read_csv(io.StringIO(flat), header=None, names=range(ncol), dtype=np.float64,
                     na_values=["NaN", "nan", ""], engine="c")
    del flat, text
    arr = df.to_numpy()
    del df

    fid = arr[:, 0].astype(np.int64)
    rows = fid - 1
    ok = (rows >= 0) & (rows < n_frames)

    P = arr[:, 1:1 + n_slots * pw].reshape(len(arr), n_slots, pw)
    B = arr[:, 1 + n_slots * pw:]

    px = np.full((n_frames, n_slots), np.nan, np.float32)
    py = np.full((n_frames, n_slots), np.nan, np.float32)
    ps = np.full((n_frames, n_slots), np.nan, np.float32)
    bx = np.full(n_frames, np.nan, np.float32)
    by = np.full(n_frames, np.nan, np.float32)

    px[rows[ok]] = P[ok, :, 0] * length
    py[rows[ok]] = P[ok, :, 1] * width
    if pw >= 3:
        s = P[ok, :, pw - 1]
        if speed_unit == "kph":
            s = s / 3.6
        ps[rows[ok]] = s
    bx[rows[ok]] = B[ok, 0] * length
    by[rows[ok]] = B[ok, 1] * width

    return {
        "px": px, "py": py, "ps": ps, "bx": bx, "by": by,
        "n_records": int(len(arr)),
        "frame_id_range": (int(fid.min()), int(fid.max())),
        "player_fields": pw, "ball_fields": bw,
    }


# --------------------------------------------------------------------------------------------
# Smart Tagging
# --------------------------------------------------------------------------------------------

def read_smart_tagging(path, fps=25):
    """[{code, start, end, tags}] with start/end as video frames (end exclusive)."""
    out = []
    root = _strip_ns(ET.parse(path).getroot())
    for inst in root.iter("instance"):
        try:
            s = float(inst.findtext("start"))
            e = float(inst.findtext("end"))
        except (TypeError, ValueError):
            continue
        tags = {}
        for lab in inst.findall("label"):
            g = (lab.findtext("group") or "").strip()
            t = (lab.findtext("text") or "").strip()
            if g:
                tags[g] = t
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(s * fps)),
            "end": int(round(e * fps)),
            "tags": tags,
        })
    return out


def team_side(team_tag, home_name, away_name):
    """'H', 'A' or None for a Smart Tagging Team value. Fuzzy on purpose: the XML says
    'Atletic Club' while the ATD may say 'Athletic Club'."""
    if not team_tag:
        return None
    t = team_tag.lower()
    if t in ("n/a", "na", "none", ""):
        return None

    def toks(s):
        return set(re.findall(r"[a-z]{4,}", (s or "").lower())) - {"club", "futbol", "football"}

    th, ta, tt = toks(home_name), toks(away_name), toks(team_tag)
    sh = len(tt & th) + sum(1 for a in tt for b in th if a[:4] == b[:4])
    sa = len(tt & ta) + sum(1 for a in tt for b in ta if a[:4] == b[:4])
    if sh > sa:
        return "H"
    if sa > sh:
        return "A"
    # Fallback for this match
    if "barcel" in t:
        return "H"
    if "tletic" in t or "athlet" in t or "bilbao" in t:
        return "A"
    return None
