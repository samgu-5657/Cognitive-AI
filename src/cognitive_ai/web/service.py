"""Workbench service layer (Phase 5A). Thin, validated wrappers over the
real cognitive modules. The web server calls THESE functions only -- never
CLI strings, never shell. All state lives in the passed directories."""
from __future__ import annotations

import os

from ..curiosity import Curiosity
from ..interact import Session
from ..learner import Learner, default_regression
from ..memory import Memory
from ..ontology import Ontology
from ..reasoner import Reasoner
from ..schema_evolver import SchemaEvolver

MAX_TEXT = 500
MAX_STEPS = 20


def _req_str(v, name, allow_empty=False) -> str:
    if not isinstance(v, str) or (not allow_empty and not v.strip()):
        raise ValueError(f"bad parameter: {name}")
    if len(v) > MAX_TEXT:
        raise ValueError(f"parameter too long: {name}")
    return v.strip()


def _req_int(v, name, lo, hi) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"bad parameter: {name}")
    if not (lo <= n <= hi):
        raise ValueError(f"parameter out of range: {name}")
    return n


class Workbench:
    def __init__(self, data_dir: str, models_root: str | None = None):
        self.data_dir = data_dir
        self.models_root = models_root or os.path.join(
            os.path.dirname(os.path.abspath(data_dir)), "models")
        os.makedirs(data_dir, exist_ok=True)
        self.mem = Memory(os.path.join(data_dir, "cognitive.db"))
        seed = os.path.join(data_dir, "concepts.json")
        if self.mem.current_version() == 0 and not self.mem.list_concepts() \
                and os.path.exists(seed):
            from ..cli import seed_from_json
            seed_from_json(self.mem, seed)
        self.onto = Ontology(self.mem)
        self.reasoner = Reasoner(self.mem)
        self.learner = Learner(self.mem,
                               os.path.join(data_dir, "corpus"),
                               enable_structural=True)
        self.curiosity = Curiosity(self.mem)
        self.sessions: dict[str, Session] = {}

    def close(self):
        self.mem.close()

    def session(self, sid: str) -> Session:
        sid = _req_str(sid or "default", "session_id")
        if sid not in self.sessions:
            self.sessions[sid] = Session(
                self.mem, os.path.join(self.data_dir, "corpus"),
                models_root=self.models_root,
                queue_path=os.path.join(self.data_dir, "queue.jsonl"))
        return self.sessions[sid]

    # ---- status ----
    def status(self) -> dict:
        from ..neural.gate import read_active
        try:
            active = read_active(self.models_root)
        except Exception:
            active = None
        last = self.mem.query(
            "SELECT task_id,task_type,status FROM cog_tasks ORDER BY "
            "rowid DESC LIMIT 1")
        return {"active_model": (active or {}).get("tag"),
                "model_record": active,
                "kb_version": SchemaEvolver(self.mem).current_version(),
                "concepts": len(self.mem.list_concepts()),
                "service": "ok",
                "last_task": last[0] if last else None}

    # ---- dialogue & solve ----
    def solve(self, text: str, sid: str, style: str = "short") -> dict:
        from ..executor import solve_text
        from ..explainer import explain
        from ..semantics import parse_sentence
        text = _req_str(text, "text")
        if style not in ("short", "steps"):
            raise ValueError("bad parameter: style")
        frame = parse_sentence(text, source="web").to_dict()
        trace = solve_text(text, frame, self.mem, self.onto, self.reasoner)
        ans = explain(trace, style=style)
        if trace["task"]["status"] != "created":
            # unparseable inputs leave zero writes, not even an episodic log
            self.mem.add_experience("evaluation", f"web solve -> "
                                    f"{trace['task']['status']}",
                                    {"task_id": trace["task"]["task_id"]})
        return {"frame": frame, "task": trace["task"], "steps": trace["steps"],
                "repairs": trace.get("repairs", []),
                "gap": trace.get("gap"),
                "failure_cause": trace.get("failure_cause"),
                "answer": ans}

    def understand(self, text: str, sid: str) -> dict:
        text = _req_str(text, "text")
        out = self.session(sid).handle(f"理解 {text}")
        return {"lines": out}

    def confirm(self, sid: str) -> dict:
        return {"lines": self.session(sid).handle("确认")}

    def chat(self, text: str, sid: str) -> dict:
        text = _req_str(text, "text", allow_empty=True)
        return {"lines": self.session(sid).handle(text or "")}

    # ---- teaching (preview + confirm, lifecycle-gated) ----
    def teach_preview(self, name: str, attributes: dict,
                      parents: list) -> dict:
        from ..novelty import NoveltyDetector
        name = _req_str(name, "name")
        if not isinstance(attributes, dict) or \
                not all(isinstance(k, str) and isinstance(v, (str, int, float, bool))
                        for k, v in attributes.items()):
            raise ValueError("bad parameter: attributes")
        if not isinstance(parents, list) or \
                not all(isinstance(p, str) for p in parents):
            raise ValueError("bad parameter: parents")
        rep = NoveltyDetector(self.mem).check_observation(
            {"name": name, "attributes": attributes, "parents": parents})
        return {"novelty": dict(rep),
                "will_create": bool(rep.get("needs_new_concept")),
                "as_status": "hypothesis (never verified directly)"}

    def teach_confirm(self, name: str, attributes: dict,
                      parents: list, source: str = "web") -> dict:
        name = _req_str(name, "name")
        source = _req_str(source or "web", "source")
        known = [p for p in (parents or [])
                 if self.mem.get_concept(p) is not None]
        unknown_parents = [p for p in (parents or []) if p not in known]
        res = self.learner.learn_from_observation(
            {"name": name, "attributes": attributes or {},
             "parents": known, "source": source})
        return {"experience_id": res.get("experience_id"),
                "novelty": res.get("novelty", {}),
                "new_concept": res.get("new_concept"),
                "updated_concept": res.get("updated_concept"),
                "unknown_parents_kept_pending": unknown_parents}

    def correct(self, name: str, key: str, value: str, reason: str) -> dict:
        name, key = _req_str(name, "name"), _req_str(key, "key")
        value, reason = _req_str(value, "value"), \
            _req_str(reason or "web correction", "reason")
        res = self.learner.learn_from_observation(
            {"name": name, "attributes": {key: value},
             "source": "web-correction"})
        self.mem.add_experience(
            "evaluation", f"web correction: {name}.{key} -> {value}",
            {"op": "correction", "reason": reason})
        return {"novelty": res.get("novelty", {}),
                "conflicts": res.get("conflicts", [])}

    # ---- rules lab (real rulelearn flow) ----
    def rule_induce(self, name: str, domain: str, examples: list,
                    negatives: list, save: bool = True) -> dict:
        from .. import rulelearn as RL
        name = _req_str(name, "name")
        if domain not in ("math", "concepts"):
            raise ValueError("bad parameter: domain")
        for rows, label in ((examples, "examples"), (negatives, "negatives")):
            if not isinstance(rows, list):
                raise ValueError(f"bad parameter: {label}")
            for r in rows:
                if not isinstance(r, dict) or "inputs" not in r \
                        or "outputs" not in r:
                    raise ValueError(f"bad example in {label}")
        rep = RL.induce(examples, negatives, domain=domain)
        out = {"candidates": rep["candidates"], "ambiguous": rep["ambiguous"],
               "note": rep["note"]}
        if save and rep["candidates"]:
            rid = RL.new_rule_id(name)
            self.mem.save_rule(RL.make_proposal_record(
                rid, name, domain, rep["candidates"],
                evidence=[{"source": "web"}]))
            out["saved"] = rid
        return out

    def rule_select(self, rule_id: str, index: int, reason: str) -> dict:
        from .. import rulelearn as RL
        row = self.mem.get_rule(_req_str(rule_id, "rule_id"))
        if not row:
            raise ValueError("rule not found")
        out = RL.select_candidate(
            row, _req_int(index, "index", 0, 99), reason or "web reviewer")
        if out.get("ok"):
            self.mem.save_rule(row)
        return out

    def rule_validate(self, rule_id: str, train: list, heldout: list,
                      negatives: list, activate: bool = False) -> dict:
        from .. import rulelearn as RL
        row = self.mem.get_rule(_req_str(rule_id, "rule_id"))
        if not row:
            raise ValueError("rule not found")
        try:
            spec = RL.selected_spec(row)
        except RL.Unsupported as e:
            return {"ok": False, "error": str(e)}
        rep = RL.validate_rule(spec, train or [], heldout or [],
                               negatives or [], self.onto)
        row["validation"] = {**rep,
                             "selected": row["spec"].get("selected")}
        self.mem.save_rule(row)
        promo = RL.promote_record(self.mem, rule_id, "validated",
                                  validation=row["validation"])
        if promo.get("ok") and activate:
            promo = RL.promote_record(self.mem, rule_id, "active")
        self.mem.add_experience("evaluation",
                                f"web rule validate {rule_id}: {promo}",
                                {"op": "rule_validate", "report": rep})
        return {"validation": rep, "promotion": promo}

    def rule_list(self, status=None, domain=None) -> list:
        if status is not None and status not in (
                "candidate", "validated", "active", "rejected",
                "superseded"):
            raise ValueError("bad parameter: status")
        return self.mem.list_rules_learned(status=status, domain=domain)

    def rule_show(self, rule_id: str) -> dict:
        row = self.mem.get_rule(_req_str(rule_id, "rule_id"))
        if not row:
            raise ValueError("rule not found")
        return row

    def rule_rollback(self, rule_id: str) -> dict:
        from .. import rulelearn as RL
        return RL.rollback_record(self.mem, _req_str(rule_id, "rule_id"))

    def rule_calls(self, rule_id: str) -> dict:
        row = self.mem.get_rule(_req_str(rule_id, "rule_id"))
        if not row:
            raise ValueError("rule not found")
        return {"calls": row["calls"], "successes": row["successes"],
                "failures": row["failures"], "status": row["status"]}

    # ---- concepts / history ----
    def concepts(self, query: str = "", kind: str = "") -> list:
        out = []
        for c in self.onto.list():
            if query and query.lower() not in c["id"].lower() \
                    and query not in c.get("name", ""):
                continue
            if kind and c["kind"] != kind:
                continue
            out.append({"id": c["id"], "name": c["name"], "kind": c["kind"],
                        "parents": self.mem.get_parents(c["id"]),
                        "children": self.mem.get_children(c["id"])})
        return out

    def concept_detail(self, cid: str) -> dict:
        return self.onto.describe(_req_str(cid, "id"))

    def history(self, limit: int = 50) -> dict:
        from ..schema_evolver import SchemaEvolver
        ev = SchemaEvolver(self.mem)
        return {"versions": ev.versions(),
                "changes": ev.history(),
                "experiences": self.mem.list_experiences(limit=limit)}

    def tasks(self, limit: int = 20) -> list:
        return self.mem.query(
            "SELECT task_id,task_type,status FROM cog_tasks ORDER BY "
            "rowid DESC LIMIT ?", (max(1, min(int(limit), 100)),))

    def task_detail(self, task_id: str) -> dict:
        from ..executor import load_trace
        tr = load_trace(self.mem, _req_str(task_id, "task_id"))
        if not tr:
            raise ValueError("task not found")
        return tr

    # ---- queue / explore / model ----
    def queue(self, status: str | None = None) -> list:
        from ..neural.queue import UnknownQueue
        if status is not None and status not in ("pending", "accepted",
                                                 "rejected"):
            raise ValueError("bad parameter: status")
        return UnknownQueue(
            os.path.join(self.data_dir, "queue.jsonl")).list(status=status)

    def queue_review(self, sample_id: str, accept: bool,
                     l1=None, l2=None) -> dict:
        from ..neural.queue import UnknownQueue
        out = UnknownQueue(
            os.path.join(self.data_dir, "queue.jsonl")).review(
                _req_str(sample_id, "sample_id"), bool(accept), l1, l2,
                reviewer="web")
        if not out.get("ok"):
            raise ValueError(out.get("error", "review failed"))
        return out

    def explore(self, steps: int = 3) -> dict:
        from ..loop import ExplorationLoop
        steps = _req_int(steps, "steps", 1, MAX_STEPS)
        return ExplorationLoop(
            self.mem, os.path.join(self.data_dir, "corpus")).run(
                steps=steps, max_seconds=60.0)

    def gaps(self) -> list:
        from ..novelty import NoveltyDetector
        return NoveltyDetector(self.mem).knowledge_gaps()

    def model_status(self) -> dict:
        from ..neural.gate import read_active
        import glob
        cands = sorted(os.path.dirname(p).split(os.sep)[-1] for p in
                       glob.glob(os.path.join(self.models_root, "*",
                                              "best.pt")))
        try:
            active = read_active(self.models_root)
        except Exception:
            active = None
        return {"active": active, "candidates": cands,
                "note": "promotion only through the neural gate; "
                        "the UI has no direct activation path"}

    def predict(self, text: str) -> dict:
        from ..neural.adapter import NeuralAdapter
        from ..neural.gate import read_active
        text = _req_str(text, "text")
        act = read_active(self.models_root) or {}
        tag = act.get("tag", "cnn-v1")
        ad = NeuralAdapter(os.path.join(self.models_root, tag, "best.pt"),
                           os.path.join(self.models_root, tag, "vocab.json"))
        out = ad.predict(text)
        try:
            from ..neural.ood import assess, load_artifacts
            arts = load_artifacts(os.path.join(self.models_root, tag))
            if arts["prototypes.json"] and arts["ood_thresholds.json"]:
                out["ood"] = assess(
                    text, ad.model, ad.vocab, arts["prototypes.json"], 1.0,
                    arts["ood_thresholds.json"])
        except Exception as e:  # noqa: BLE001 - OOD is best-effort
            out["ood"] = {"verdict": "unavailable", "reasons": [str(e)]}
        return out
