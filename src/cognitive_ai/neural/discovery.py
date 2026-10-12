"""Candidate concept discovery.

Operates on REVIEWED samples (human labels) + real model features.
Never on bare network predictions, never on name similarity.

For each reviewer-L1 group with n >= min_n:
  cohesion    mean pairwise cosine similarity of sample features;
  explained   fraction whose nearest L1 prototype is the reviewed L1
              within dist_far;
  l1_gap      centroid distance to the reviewed-L1 prototype / spread;
  l2_gap      centroid distance to the nearest existing L2 centroid.

Decisions:
  existing_ok         explained >= 2/3 (current categories suffice);
  propose new_l2      reviewer L2 is 'none', group cohesive, explained by
                      L1, but far from every existing L2 centroid;
  propose new_l1      cohesive group far from ALL L1 prototypes
                      (candidate PROPOSAL only, never auto-applied);
  insufficient_evidence  too few samples, dispersed, or contradictory.

Validation (validate_proposal): leave-one-out nearest-centroid check --
a held-out member must land nearer its group centroid than any existing
centroid in >= 75% of rounds, else the proposal is REJECTED. No training
involved, deterministic.

Small-sample honesty: cohesion bar is a documented heuristic (0.5); the
LOO gate is the binding check.
"""
from __future__ import annotations

import math

COHESION_MIN = 0.5
EXPLAINED_MIN = 2 / 3
LOO_MIN = 0.75


def _cos(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mean(vecs):
    return [sum(c) / len(vecs) for c in zip(*vecs)]


def l2_prototypes(model, vocab, train_rows: list[dict]) -> dict:
    """Mean feature per L2 subcategory (n>=2) + family + spread.

    Train rows only. 'family' is the L1 class of the member rows.
    """
    from .dataset import labeled
    torch = _torch()
    rows = [r for r in labeled(train_rows)
            if (r.get("level2") or "none") != "none"]
    if not rows:
        return {}
    feats = model.features(
        torch.tensor([vocab.encode(r["text"]) for r in rows])).tolist()
    by_l2: dict[str, list] = {}
    fam: dict[str, str] = {}
    for r, f in zip(rows, feats):
        by_l2.setdefault(r["level2"], []).append(f)
        fam[r["level2"]] = r["level1"]
    out = {}
    for k, v in by_l2.items():
        if len(v) < 2:
            continue
        mean = _mean(v)
        out[k] = {"mean": mean, "n": len(v), "family": fam[k],
                  "spread": sum(_dist(f, mean) for f in v) / len(v)}
    return out


def family_gap(centroid, l2_protos: dict, family: str,
               dist_far: float) -> tuple:
    """Min centroid distance to same-family L2 centroids + whether all
    exceed the novelty radius. Cross-family centroids are ignored: a
    kitchenware cluster is trivially 'far' from emotion in raw units."""
    ds = sorted(_dist(centroid, p["mean"])
                for k, p in l2_protos.items() if p["family"] == family)
    if not ds:
        return None, True
    return round(ds[0], 3), all(d > dist_far for d in ds)


def _torch():
    from . import require_torch
    return require_torch()


def _features_of(model, vocab, texts):
    """Encode texts unless the model takes raw inputs (test stubs)."""
    torch = _torch()
    if vocab is None:
        return model.features(texts)
    return model.features(
        torch.tensor([vocab.encode(t) for t in texts]))


def discover(samples: list[dict], model, vocab, l1_protos: dict,
             l2_protos: dict, thresholds: dict,
             min_n: int = 3) -> dict:
    """Propose concept structure (or explicitly decline). Pure function."""
    torch = _torch()
    if not samples:
        return {"kind": "insufficient_evidence", "reason": "no samples"}
    feats = _features_of(model, vocab,
                         [s["text"] for s in samples]).tolist()
    by_l1: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_l1.setdefault(s["l1"], []).append(i)
    proposals = []
    flags = []
    for l1, idx in sorted(by_l1.items()):
        if len(idx) < min_n:
            proposals.append({"kind": "insufficient_evidence", "l1": l1,
                              "n": len(idx),
                              "reason": f"need >= {min_n} samples"})
            continue
        vecs = [feats[i] for i in idx]
        pairs = [_cos(vecs[a], vecs[b])
                 for a in range(len(vecs)) for b in range(a + 1, len(vecs))]
        cohesion = sum(pairs) / len(pairs) if pairs else 1.0
        centroid = _mean(vecs)
        proto = l1_protos.get(l1)
        explained = 0
        disagree = []
        d_l1min, d_l1min_name = None, None
        for i in idx:
            ds = sorted(((n, _dist(feats[i], p["mean"]))
                         for n, p in l1_protos.items()),
                        key=lambda kv: kv[1])
            if ds[0][0] == l1 and ds[0][1] <= thresholds["dist_far"]:
                explained += 1
            else:
                disagree.append(samples[i].get("sample_id", f"#{i}"))
            if d_l1min is None or ds[0][1] < d_l1min:
                d_l1min, d_l1min_name = ds[0][1], ds[0][0]
        if len(disagree) >= 2:
            # neural-side misplacement signal: reviewer label repeatedly
            # loses to another prototype. Evidence only -- converting this
            # into an ontology reparent needs a reviewer-supplied concept id.
            flags.append({"type": "misplaced_suspect", "l1": l1,
                          "nearest": d_l1min_name,
                          "sample_ids": disagree})
        explained_frac = explained / len(idx)
        l2_none = all((samples[i].get("l2") or "none") == "none"
                      for i in idx)
        d_fam, fam_clear = family_gap(centroid, l2_protos, l1,
                                      thresholds["dist_far"])
        spread = (proto or {}).get("spread", 1.0) or 1.0
        base = {"l1": l1, "n": len(idx),
                "cohesion": round(cohesion, 3),
                "explained": round(explained_frac, 3),
                "family_gap": d_fam,
                "sample_ids": [samples[i].get("sample_id", f"#{i}")
                               for i in idx]}
        if explained_frac >= EXPLAINED_MIN and not (l2_none and fam_clear):
            proposals.append({**base, "kind": "existing_ok",
                              "reason": "current categories explain the group"})
        elif l2_none and cohesion >= COHESION_MIN and fam_clear:
            # Reviewer filed these under an L1 with no fitting L2; the group
            # is cohesive and far from every existing L2 centroid. The L1
            # disagreement (if any) is kept as a caveat + misplaced flag;
            # the reviewer label decides the parent, not the network.
            proposals.append({
                **base, "kind": "new_l2",
                "label": f"{l1}_cand{len([p for p in proposals if p.get('kind') == 'new_l2']) + 1}",
                "l1_explained": round(explained_frac, 3),
                "reason": f"cohesive (c={cohesion:.2f}) under reviewer L1 "
                          f"'{l1}', clear of all same-family L2 centroids "
                          f"(gap={d_fam})"})
        elif (cohesion >= COHESION_MIN and explained_frac < 0.5
                and d_l1min is not None
                and d_l1min > thresholds["dist_far"]):
            proposals.append({
                **base, "kind": "new_l1_candidate",
                "reason": f"cohesive but nearest L1 is {d_l1min_name} "
                          f"at d={d_l1min:.2f} (proposal only)"})
        else:
            proposals.append({**base, "kind": "insufficient_evidence",
                              "reason": "dispersed or contradictory evidence"})
    # overall: surface the strongest actionable proposal, if any
    rank = {"new_l2": 0, "new_l1_candidate": 1, "existing_ok": 2,
            "insufficient_evidence": 3}
    proposals.sort(key=lambda p: rank.get(p["kind"], 9))
    return {"proposals": proposals, "selected": proposals[0], "flags": flags}


def validate_proposal(samples: list[dict], model, vocab, l1_protos: dict,
                      l2_protos: dict, min_rate: float = LOO_MIN) -> dict:
    """Leave-one-out centroid check. No training, deterministic."""
    torch = _torch()
    if len(samples) < 2:
        return {"ok": False, "reason": "need >= 2 samples"}
    feats = _features_of(model, vocab,
                         [s["text"] for s in samples]).tolist()
    wins = 0
    for i in range(len(feats)):
        others = [f for j, f in enumerate(feats) if j != i]
        c = _mean(others)
        rivals = [_dist(feats[i], p["mean"])
                  for p in list(l1_protos.values())
                  + list(l2_protos.values())]
        if _dist(feats[i], c) < (min(rivals) if rivals else float("inf")):
            wins += 1
    rate = wins / len(feats)
    return {"ok": rate >= min_rate, "rate": round(rate, 3),
            "wins": wins, "n": len(feats)}


def to_schema_payload(proposal: dict, extra: dict | None = None) -> dict:
    """Convert split/misplaced proposals to SchemaEvolver payloads."""
    kind = proposal.get("kind")
    if kind == "split":
        return {"kind": "split_category", "payload": proposal["payload"],
                "reason": proposal.get("reason", ""),
                "evidence": (extra or {}).get("evidence", [])}
    if kind == "misplaced":
        return {"kind": "reparent",
                "payload": {"concept": proposal["concept"],
                            "parent": proposal["new_parent"]},
                "reason": proposal.get("reason", ""),
                "evidence": (extra or {}).get("evidence", [])}
    raise ValueError(f"no schema mapping for proposal kind {kind!r}")


def apply_new_l2(base_dir: str, out_dir: str, new_label: str,
                 base_map: list[str] | None = None) -> dict:
    """Materialize a validated new-L2 proposal as a candidate model dir.

    Copies the base vocab, extends the L2 head (old rows exact), saves
    candidate.pt + meta. Does NOT touch the active pointer; training with
    the new label happens via incremental_train(l2_labels=[...]), and
    promotion stays behind the model gate. L1-new proposals have no path
    here by design (candidate claims only).
    """
    import json
    import os
    import shutil
    from . import LEVEL2
    torch = _torch()
    from .trainer import extend_l2, load_model, weights_fingerprint
    if base_map is None:
        base_map = list(LEVEL2)
    if new_label in base_map:
        return {"ok": False, "error": f"{new_label} already mapped"}
    os.makedirs(out_dir, exist_ok=True)
    model, vocab, ck = load_model(os.path.join(base_dir, "best.pt"),
                                  os.path.join(base_dir, "vocab.json"))
    before = weights_fingerprint(model)
    model, full = extend_l2(model, base_map, base_map + [new_label])
    shutil.copy(os.path.join(base_dir, "vocab.json"),
                os.path.join(out_dir, "vocab.json"))
    torch.save({"kind": ck["kind"], "seed": ck.get("seed"),
                "max_len": vocab.max_len,
                "hparams": ck.get("hparams", {}),
                "level1": ck.get("level1", None) or _l1(),
                "level2": full, "base": base_dir,
                "state": model.state_dict()},
               os.path.join(out_dir, "candidate.pt"))
    meta = {"base": base_dir, "new_label": new_label,
            "before": before,
            "after": weights_fingerprint(model)}
    with open(os.path.join(out_dir, "extended.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    meta.update({"ok": True, "out_dir": out_dir,
                 "checkpoint": os.path.join(out_dir, "candidate.pt")})
    return meta


def _l1():
    from . import LEVEL1
    return LEVEL1
