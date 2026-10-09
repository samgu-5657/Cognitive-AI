"""Learnable rules: induction, interpretation, validation (Phase 4D).

Two bounded rule families (closed DSL, no eval, no codegen):
  slot      preconditions over input slots -> emit a label/attribute.
            {pre:[{pred,slot,value}], out:{kind,label|attr}}
  transform scalar input -> scalar output via bind/check/compute/emit steps.

Statuses: candidate -> validated -> active; rejected at any point;
superseded on replacement. Only ACTIVE rules run in the executor, and
only through the interpreter below. Promotion requires an independent
validation suite (train rows are never the validation set).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from fractions import Fraction


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


STATUSES = ("candidate", "validated", "active", "rejected", "superseded")


def new_rule_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "rule"
    return f"lr_{slug}"


def make_spec(kind: str, name: str, domain: str, inputs: dict,
              pre: list, out: dict, scope: dict | None = None,
              positives: list | None = None,
              negatives: list | None = None, evidence: list | None = None,
              steps: list | None = None) -> dict:
    """Structured spec constructor. No prose-only rules possible."""
    if kind not in ("slot", "transform"):
        raise ValueError(f"unsupported rule kind {kind!r}")
    return {"kind": kind, "name": name, "domain": domain,
            "inputs": inputs, "pre": pre or [], "out": out,
            "steps": steps or [],
            "scope": scope or {"applies_to": [domain], "exceptions": []},
            "positives": positives or [], "negatives": negatives or [],
            "evidence": evidence or []}


def new_record(rule_id: str, spec: dict) -> dict:
    """A proposal record wraps one or more candidate specs.

    Single-candidate proposals auto-select; ambiguous ones MUST be
    resolved by select_candidate() (human or evidence-driven) before
    any promotion: the system never silently picks among rivals.
    """
    cands = spec.get("candidates") or [{"form": "single", "spec": spec}]
    return {"id": rule_id, "version": 1,
            "name": spec.get("name", rule_id),
            "domain": spec.get("domain", ""),
            "spec": {"candidates": cands,
                     "selected": 0 if len(cands) == 1 else None,
                     "ambiguous": len(cands) > 1},
            "status": "candidate", "validation": {}, "calls": 0,
            "successes": 0, "failures": 0, "previous": "",
            "created_at": _now()}


def make_proposal_record(rule_id: str, name: str, domain: str,
                         candidates: list, evidence: list | None = None
                         ) -> dict:
    return new_record(rule_id, {"name": name, "domain": domain,
                                "candidates": candidates,
                                "evidence": evidence or []})


def select_candidate(record: dict, index: int, reason: str,
                     by: str = "reviewer") -> dict:
    cands = record["spec"].get("candidates", [])
    if not (0 <= index < len(cands)):
        return {"ok": False, "error": "candidate index out of range"}
    record["spec"]["selected"] = index
    record["spec"]["selection"] = {"by": by, "reason": reason,
                                   "at": _now()}
    return {"ok": True, "selected": cands[index].get("form")}


def selected_spec(record: dict) -> dict:
    idx = record["spec"].get("selected")
    cands = record["spec"].get("candidates", [])
    if idx is None or not (0 <= idx < len(cands)):
        raise Unsupported("no candidate selected; resolve ambiguity first")
    return cands[idx]["spec"]


# ---------------------------------------------------------------- interpreter
class Unsupported(Exception):
    pass


def _num(v):
    if isinstance(v, bool):
        raise Unsupported("booleans are not arithmetic operands")
    if isinstance(v, (int, Fraction)):
        return Fraction(v)
    if isinstance(v, float):
        return Fraction(v).limit_denominator(10 ** 6)
    if isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"-?\d+", s):
            return Fraction(int(s))
        if re.fullmatch(r"-?\d+/\d+", s):
            n, d = s.split("/")
            return Fraction(int(n), int(d))
    raise Unsupported(f"not a number: {v!r}")


def _arith(op: str, a: Fraction, b: Fraction) -> Fraction:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise Unsupported("division by zero")
        return a / b
    if op == "^":
        if b.denominator != 1 or b < -100 or b > 100:
            raise Unsupported("exponent out of range")
        return a ** int(b)
    if op == "mod":
        if b == 0:
            raise Unsupported("modulo by zero")
        return Fraction(int(a) % int(b))
    raise Unsupported(f"unknown arithmetic op {op!r}")


def _resolve_arg(arg, bindings):
    if isinstance(arg, str) and arg.startswith("$"):
        if arg[1:] not in bindings:
            raise Unsupported(f"unbound variable {arg!r}")
        return bindings[arg[1:]]
    return arg


def _check_pred(pred: str, args: list, bindings: dict, ontology=None) -> bool:
    def val(x):
        return _resolve_arg(x, bindings)
    if pred == "equals":
        return val(args[0]) == val(args[1])
    if pred == "matches":
        return re.fullmatch(args[1], str(val(args[0]))) is not None
    if pred in ("gt", "lt", "ge", "le"):
        a, b = _num(val(args[0])), _num(val(args[1]))
        return {"gt": a > b, "lt": a < b, "ge": a >= b, "le": a <= b}[pred]
    if pred == "in_category":
        if ontology is None:
            raise Unsupported("in_category needs an ontology")
        return ontology.is_a(str(val(args[0])), str(args[1]))
    if pred == "has_attr":
        if ontology is None:
            raise Unsupported("has_attr needs an ontology")
        inh = ontology.inherited_attributes(str(val(args[0])))
        if len(args) > 2:
            return args[1] in inh and inh[args[1]]["value"] == args[2]
        return args[1] in inh
    raise Unsupported(f"unknown predicate {pred!r}")


def run_rule(spec: dict, inputs: dict, ontology=None) -> dict:
    """Execute a LearnableRule spec. Returns {status, output, trace} with
    status ok | pre_failed | unsupported. Never raises on bad input."""
    trace = []
    try:
        if spec.get("kind") == "slot":
            bindings = dict(inputs)
            for cond in spec.get("pre", []):
                ok = _check_pred(cond["pred"], cond.get("args", []),
                                 bindings, ontology)
                trace.append({"check": cond, "result": ok})
                if not ok:
                    return {"status": "pre_failed", "output": None,
                            "trace": trace}
            out = spec.get("out", {})
            return {"status": "ok",
                    "output": {"kind": out.get("kind"),
                               "label": out.get("label"),
                               "attr": out.get("attr"),
                               "value": out.get("value")},
                    "trace": trace}
        if spec.get("kind") == "transform":
            bindings: dict = {}
            for step in spec.get("steps", []):
                op = step.get("op")
                if op == "bind":
                    if step["from"] not in inputs:
                        return {"status": "pre_failed", "output": None,
                                "trace": trace + [
                                    {"bind": step, "result": "missing"}]}
                    bindings[step["var"]] = inputs[step["from"]]
                    trace.append({"bind": step, "result": "ok"})
                elif op == "check":
                    ok = _check_pred(step["pred"], step.get("args", []),
                                     bindings, ontology)
                    trace.append({"check": step, "result": ok})
                    if not ok:
                        return {"status": "pre_failed", "output": None,
                                "trace": trace}
                elif op == "compute":
                    a = _num(_resolve_arg(step["args"][0], bindings))
                    b = _num(_resolve_arg(step["args"][1], bindings))
                    bindings[step["into"]] = _arith(step["arith"], a, b)
                    trace.append({"compute": step,
                                  "result": str(bindings[step["into"]])})
                elif op == "emit":
                    tmpl = step.get("template", "$out")
                    val = bindings.get(tmpl[1:], tmpl) \
                        if isinstance(tmpl, str) and tmpl.startswith("$") \
                        else tmpl
                    if isinstance(val, Fraction):
                        val = str(val.numerator) if val.denominator == 1 \
                            else f"{val.numerator}/{val.denominator}"
                    return {"status": "ok", "output": val, "trace": trace}
                else:
                    raise Unsupported(f"unknown step op {op!r}")
            return {"status": "pre_failed", "output": None, "trace": trace}
        raise Unsupported(f"unknown rule kind {spec.get('kind')!r}")
    except Unsupported as e:
        return {"status": "unsupported", "output": None,
                "trace": trace + [{"unsupported": str(e)}]}


# --------------------------------------------------------------- induction
def _slot_values(examples: list[dict]) -> dict:
    vals: dict[str, set] = {}
    for e in examples:
        for k, v in e["inputs"].items():
            vals.setdefault(k, set()).add(_freeze(v))
    return vals


def _freeze(v):
    if isinstance(v, (dict, list)):
        import json
        return json.dumps(v, sort_keys=True)
    return v


def _covers_slot(rule_pre: list, example: dict, ontology=None) -> bool:
    bindings = dict(example["inputs"])
    try:
        return all(_check_pred(c["pred"], c.get("args", []), bindings,
                               ontology) for c in rule_pre)
    except Unsupported:
        return False


def _same_out(a, b) -> bool:
    return _freeze(a) == _freeze(b)


def propose_affine(examples: list[dict], in_slot="n",
                   out_slot="out") -> list[dict]:
    """Fit output = a*input + b exactly (rational). Needs >= 2 examples."""
    pts = []
    for e in examples:
        try:
            pts.append((_num(e["inputs"][in_slot]), _num(e["outputs"][out_slot])))
        except (KeyError, Unsupported):
            return []
    if len(pts) < 2:
        return []
    (x0, y0), (x1, y1) = pts[0], pts[1]
    if x1 == x0:
        return []
    a = (y1 - y0) / (x1 - x0)
    b = y0 - a * x0
    if not all(y == a * x + b for x, y in pts):
        return []
    fa = f"{a.numerator}/{a.denominator}" if a.denominator != 1 else str(a.numerator)
    fb = f"{b.numerator}/{b.denominator}" if b.denominator != 1 else str(b.numerator)
    return [{"form": f"out = {fa}*n + {fb}",
             "constant": (a == 0),  # constant output = memorization risk;
             # the held-out suite (not this flag) decides its fate
             "spec": make_spec(
                 "transform", f"affine_{fa}_{fb}", "math",
                 {"n": "number"}, [], {"kind": "value"},
                 steps=[{"op": "bind", "var": "n", "from": in_slot},
                        {"op": "compute", "arith": "*", "args": ["$n", fa],
                         "into": "t"},
                        {"op": "compute", "arith": "+", "args": ["$t", fb],
                         "into": "o"},
                        {"op": "emit", "template": "$o"}],
                 positives=[e["id"] if "id" in e else "" for e in examples])}]


def propose_modular(examples: list[dict], in_slot="n", out_slot="out",
                    mod: int = 2) -> list[dict]:
    """Output class from input mod m. All examples must share their
    residue->class mapping consistently, and every mapped residue needs
    >= 2 supporting examples -- one example per residue is memorization,
    not a rule, and is rejected here."""
    mapping = {}
    counts: dict[int, int] = {}
    for e in examples:
        try:
            r = int(_num(e["inputs"][in_slot])) % mod
        except (KeyError, Unsupported):
            return []
        cls = _freeze(e["outputs"][out_slot])
        if r in mapping and mapping[r] != cls:
            return []
        mapping[r] = cls
        counts[r] = counts.get(r, 0) + 1
    if not mapping or min(counts.values()) < 2:
        return []
    steps = [{"op": "bind", "var": "n", "from": in_slot},
               {"op": "compute", "arith": "mod",
                "args": ["$n", str(mod)], "into": "r"}]
    for res, cls in sorted(mapping.items()):
        steps.append({"op": "check", "pred": "equals",
                      "args": ["$r", str(res)]})
        steps.append({"op": "emit", "template": cls})
    return [{"form": f"class by n mod {mod}: {mapping}",
             "spec": make_spec(
                 "transform", f"mod{mod}_class", "math",
                 {"n": "number"}, [], {"kind": "class"},
                 steps=steps,
                 positives=[e.get("id", "") for e in examples])}]


def _modular_run(spec: dict, inputs: dict) -> dict:
    """Chain semantics for modular specs: first matching residue emits."""
    trace = []
    bindings: dict = {}
    residue = None
    for step in spec.get("steps", []):
        if step["op"] == "bind":
            if step["from"] not in inputs:
                return {"status": "pre_failed", "output": None, "trace": trace}
            bindings[step["var"]] = inputs[step["from"]]
        elif step["op"] == "compute":
            a = _num(_resolve_arg(step["args"][0], bindings))
            b = _num(_resolve_arg(step["args"][1], bindings))
            residue = str(_arith(step["arith"], a, b))
            trace.append({"residue": residue})
        elif step["op"] == "check":
            if str(residue) == str(step["args"][1]):
                trace.append({"matched_residue": residue})
                # next step must be its emit
                continue
            trace.append({"skipped_residue": step["args"][1]})
        elif step["op"] == "emit":
            # emit only if the immediately preceding check matched
            if trace and "matched_residue" in trace[-1]:
                return {"status": "ok", "output": step["template"],
                        "trace": trace}
    return {"status": "pre_failed", "output": None, "trace": trace}


def propose_conjunction(positives: list[dict], negatives: list[dict],
                        label: str) -> list[dict]:
    """Greedy minimal slot=value conjunction covering all positives and no
    negatives. Empty result = no consistent conjunction exists."""
    if not positives:
        return []
    shared = set(positives[0]["inputs"].items())
    for e in positives[1:]:
        shared &= set(e["inputs"].items())
    shared = sorted(shared)
    # greedy set cover: at each step take the shared attribute excluding
    # the most remaining negatives (a condition matching every negative
    # explains nothing and is skipped while danger remains)
    chosen = []
    remaining_negs = list(negatives)
    unused = list(shared)
    while remaining_negs and unused:
        scored = []
        for k, v in unused:
            excluded = sum(1 for n in remaining_negs
                           if n["inputs"].get(k) != v)
            matched_pos = sum(1 for e in positives
                              if e["inputs"].get(k) == v)
            scored.append((excluded, matched_pos, k, v))
        scored.sort(reverse=True)
        best_excluded, _, bk, bv = scored[0]
        if best_excluded <= 0:
            break
        chosen.append({"pred": "equals", "args": [f"${bk}", bv]})
        unused = [(k, v) for k, v in unused if not (k == bk and v == bv)]
        remaining_negs = [n for n in remaining_negs
                          if n["inputs"].get(bk) == bv]
    if remaining_negs:
        return []  # shared attrs cannot exclude every negative
    if not chosen:
        # no negatives to beat: pin the first shared attribute so the
        # rule is never vacuous
        k, v = shared[0]
        chosen = [{"pred": "equals", "args": [f"${k}", v]}]
    return [{"form": " AND ".join(f"{c['args'][0]}={c['args'][1]!r}"
                                  for c in chosen),
             "spec": make_spec(
                 "slot", f"conj_{label}", "concepts",
                 {k: "any" for k, _ in shared}, chosen,
                 {"kind": "label", "label": label},
                 positives=[e.get("id", "") for e in positives],
                 negatives=[e.get("id", "") for e in negatives])}]


def induce(examples: list[dict], negatives: list[dict] | None = None,
           domain: str = "math", in_slot="n",
           out_slot="out") -> dict:
    """Propose candidate rules from examples. Returns ALL consistent
    candidates with coverage -- ambiguity is reported, never hidden."""
    negatives = negatives or []
    cands = []
    if domain == "math":
        cands += [{"proposer": "affine", **c}
                  for c in propose_affine(examples, in_slot, out_slot)]
        cands += [{"proposer": "modular", **c}
                  for c in propose_modular(examples, in_slot, out_slot)]
    else:
        labels = {}
        for e in examples:
            labels.setdefault(_freeze(e["outputs"].get("label")), []).append(e)
        for label, pos in labels.items():
            for c in propose_conjunction(pos, negatives, label):
                cands.append({"proposer": "conjunction", **c})
    # consistency filter: candidate must cover every positive and no negative
    kept = []
    for c in cands:
        if all(_covers_candidate(c["spec"], e) for e in examples) and \
                not any(_covers_candidate(c["spec"], n) for n in negatives):
            kept.append(c)
    return {"candidates": kept,
            "ambiguous": len(kept) > 1,
            "note": "multiple consistent rules; supply counterexamples"
                    if len(kept) > 1 else ""}


def _is_modular(spec: dict) -> bool:
    return any(s.get("op") == "compute" and s.get("arith") == "mod"
               for s in spec.get("steps", []))


def apply_spec(spec: dict, inputs: dict, ontology=None) -> dict:
    """Single execution entry: modular chain or generic interpreter."""
    if spec.get("kind") == "transform" and _is_modular(spec):
        return _modular_run(spec, inputs)
    return run_rule(spec, inputs, ontology)


def _covers_candidate(spec: dict, example: dict, ontology=None) -> bool:
    if spec["kind"] == "slot":
        r = run_rule(spec, example["inputs"], ontology)
        if r["status"] != "ok" or r["output"] is None:
            return False
        want = example["outputs"].get("label")
        return r["output"].get("label") == want
    if spec["kind"] == "transform":
        r = apply_spec(spec, example["inputs"], ontology)
        if r["status"] != "ok":
            return False
        try:
            return _num(r["output"]) == _num(example["outputs"]["out"])
        except (KeyError, Unsupported):
            return _freeze(r["output"]) == \
                _freeze(example["outputs"].get("out",
                                               example["outputs"].get("label")))
    return False


# -------------------------------------------------------------- validation
def validate_rule(spec: dict, train_rows: list[dict],
                  heldout_rows: list[dict], negatives: list[dict],
                  ontology=None) -> dict:
    """Independent suite: train refit, held-out, negatives, invalid input.

    Returns {passed, checks{...}}. Train rows must hit 100% (they induced
    the rule); held-out decides generalization; negatives must be refused;
    missing slots must yield pre_failed/unsupported, never a crash.
    """
    def acc(rows):
        if not rows:
            return None
        ok = sum(1 for e in rows if _covers_candidate(spec, e, ontology))
        return round(ok / len(rows), 4)

    neg_ok = 0
    for n in negatives:
        # a negative must NOT trigger the rule at all (firing with any
        # output is a failure; matching the neg's own label is irrelevant)
        try:
            fires = apply_spec(spec, n["inputs"], ontology)["status"] == "ok"
        except Exception:
            fires = False
        neg_ok += (not fires)
    bad_input = run_rule(spec, {}, ontology)
    checks = {
        "train_acc": acc(train_rows),
        "heldout_acc": acc(heldout_rows),
        "negatives_refused": f"{neg_ok}/{len(negatives)}" if negatives
        else "n/a",
        "invalid_input_safe": bad_input["status"] in ("pre_failed",
                                                      "unsupported"),
    }
    passed = (checks["train_acc"] == 1.0
              and (checks["heldout_acc"] or 0) >= 0.75
              and neg_ok == len(negatives)
              and checks["invalid_input_safe"])
    return {"passed": passed, "checks": checks}


def promote_record(mem, rid: str, to_status: str,
                   validation: dict | None = None) -> dict:
    """Candidate lifecycle with version history + rollback support.

    Legal transitions: candidate->validated->active, any->rejected,
    rejected->candidate, active->superseded. Real transitions bump the
    version, snapshot `previous`, and record a lifecycle event.
    Idempotent repeats are explicit no-ops (no version bump, no history
    rewrite): re-validating a validated rule refreshes its report;
    re-activating an active rule does nothing. Anything else is an
    error, never a silent state change.
    """
    if to_status not in STATUSES:
        return {"ok": False, "error": f"bad status {to_status!r}"}
    row = mem.get_rule(rid)
    if row is None:
        return {"ok": False, "error": "rule not found"}
    if to_status == row["status"]:
        return _repeat_promotion(mem, row, validation)
    allowed = {"candidate": ("validated", "rejected"),
               "validated": ("active", "rejected"),
               "active": ("superseded", "rejected"),
               "rejected": ("candidate",),
               "superseded": ()}
    if to_status not in allowed.get(row["status"], ()):
        return {"ok": False,
                "error": f"{row['status']} -> {to_status} not allowed"}
    if to_status in ("validated", "active"):
        if not (validation or row.get("validation") or {}).get("passed"):
            return {"ok": False,
                    "error": "promotion needs a passed validation suite"}
        if (row.get("spec") or {}).get("selected") is None:
            return {"ok": False,
                    "error": "ambiguous proposal: select a candidate first"}
    prev = {"version": row["version"], "spec": row["spec"],
            "status": row["status"]}
    row["previous"] = prev
    row["version"] = row["version"] + 1
    row["status"] = to_status
    if validation is not None:
        row["validation"] = validation
    mem.save_rule(row)
    mem.record_lifecycle("rule", rid, prev["status"], to_status,
                         f"rule {to_status}", [validation or {}])
    return {"ok": True, "version": row["version"]}


def _repeat_promotion(mem, row: dict, validation: dict | None) -> dict:
    """Idempotent repeat of the current status. No version bump, no
    previous/history rewrite -- a repeat is not a transition."""
    status = row["status"]
    if status == "validated":
        report = validation or {}
        if report and not report.get("passed"):
            return {"ok": False,
                    "error": "re-validation failed; status unchanged",
                    "version": row["version"]}
        if report:
            row["validation"] = report
            mem.save_rule(row)
        return {"ok": True, "version": row["version"],
                "note": "already validated; report refreshed, "
                        "version unchanged"}
    if status == "active":
        return {"ok": True, "version": row["version"],
                "note": "already active; no-op, version unchanged"}
    return {"ok": False,
            "error": f"{status} -> {status} not allowed"}


def apply_validation_outcome(mem, rid: str, report: dict,
                             activate: bool = False) -> dict:
    """Shared validate(+activate) sequencing for UI, CLI and API.

    - candidate + passed  -> validated (then active if asked);
    - validated + passed  -> report refresh, then active if asked;
    - active + passed     -> stays active (re-validated, no-op);
    - failed report       -> status unchanged, promotion refused.
    Never demotes, never force-advances, never bumps versions on repeats.
    """
    row = mem.get_rule(rid)
    if row is None:
        return {"ok": False, "error": "rule not found"}
    if not report.get("passed"):
        return {"ok": False,
                "error": "validation failed; "
                         f"status unchanged ({row['status']})"}
    status = row["status"]
    if status == "candidate":
        out = promote_record(mem, rid, "validated",
                             validation={**report, **_sel(row)})
        if not out.get("ok"):
            return out
    elif status == "validated":
        out = promote_record(mem, rid, "validated",
                             validation={**report, **_sel(row)})
        if not out.get("ok"):
            return out
    elif status != "active":
        return {"ok": False,
                "error": f"cannot validate from status {status!r}"}
    if activate and mem.get_rule(rid)["status"] != "active":
        out = promote_record(mem, rid, "active")
        if not out.get("ok"):
            return out
    final = mem.get_rule(rid)
    return {"ok": True, "version": final["version"],
            "status": final["status"]}


def _sel(row: dict) -> dict:
    sel = (row.get("spec") or {}).get("selected")
    return {"selected": sel} if sel is not None else {}


def rollback_record(mem, rid: str) -> dict:
    row = mem.get_rule(rid)
    if row is None or not row.get("previous"):
        return {"ok": False, "error": "nothing to roll back to"}
    prev = row["previous"]
    row["spec"] = prev["spec"]
    row["status"] = prev["status"]
    row["version"] = row["version"] + 1
    row["previous"] = ""
    mem.save_rule(row)
    mem.record_lifecycle("rule", rid, "rollback", row["status"],
                         "rolled back", [])
    return {"ok": True, "status": row["status"]}
