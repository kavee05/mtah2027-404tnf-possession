# More than a Hack 2027 — Phase 1: Ball Possession

Team **404_TNF**. Produces one `H` / `A` / `D` / `X` label per frame of the tactical video
(199,319 rows) for FC Barcelona vs Athletic Club, from the provided automatic tracking (ATD)
and Smart Tagging.

**Approach in one line:** per-frame evidence (ball proximity, a self-supervised team model, and
dead-ball cues) decoded with a 3-state Viterbi that charges a cost for every possession change,
so the output is whole possessions rather than flickering frames — which is what `spellF1`
rewards. Details in [TECHNICAL.md](TECHNICAL.md).

## Reproduce

```bash
python -m venv .venv   # Python 3.11+ && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # numpy + pandas only

# put the match bundle in ./data (data/tactical/BAR-ATH_tactical_atd.xml, ...),
# or point MTAH_DATA_DIR at it
python -m possession.pipeline --out submissions/404_TNF.csv
python evaluation/validate_submission.py submissions/404_TNF.csv
```

The pipeline is deterministic (no random seeds, no training randomness), so this regenerates the
submitted file byte-for-byte. It takes well under a minute on a laptop.

Only three files from the bundle are read: `BAR-ATH_tactical_atd.xml`,
`BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_smart_tagging.xml`. Videos, homography and the
broadcast recording are not used.

Optional sanity check against the public 5 minutes (used **only** to check that frames are aligned
and labels mean what we think — never for tuning, per RULES.md):

```bash
cp submissions/404_TNF.csv evaluation/results/
python evaluation/run_evaluation.py
```

## Layout

```
possession/
  data.py       ATD metadata/raw + Smart Tagging readers (numpy/pandas/stdlib)
  features.py   ball kinematics, nearest player per team, team shape, tag indicators
  model.py      logistic regression, 2-component GMM, 3-state Viterbi
  pipeline.py   end-to-end: data -> evidence -> decode -> CSV (+ diagnostics JSON)
games.py        per-match constants, copied unchanged from the challenge repository
evaluation/     official scorer + validator + public GT, copied unchanged (MIT)
TECHNICAL.md    the technical document
```

The match data is **not** in this repository (`data/` is gitignored) and must not be committed.

## Licence

Our code: MIT. `games.py` and `evaluation/` are from
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
under their MIT licence (`LICENSE-challenge`).
