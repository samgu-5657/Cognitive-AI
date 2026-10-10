"""Toy micro-world (Phase 6A, Task 6). A tiny, EXTENSIBLE environment:
objects with states, two actions, deterministic dynamics, scripted events.

The dynamics table below is the SIMULATOR (environment physics). The
learner never sees it: it only receives (state, action) examples,
feedback and counterexamples, and must induce the same transitions.
Adding objects/states/actions extends the world without touching the
learner -- that is the anti-hardcoding guarantee under test.
"""
from __future__ import annotations

STATES = ("固态", "液态", "气态")
ACTIONS = ("heat", "cool")

# Hidden physics: (state, action) -> new state.
DYNAMICS = {
    ("固态", "heat"): "液态",
    ("液态", "heat"): "气态",
    ("气态", "heat"): "气态",
    ("气态", "cool"): "液态",
    ("液态", "cool"): "固态",
    ("固态", "cool"): "固态",
}

ACTION_ZH = {"heat": "加热", "cool": "冷却"}


class ToyWorld:
    def __init__(self, objects: dict[str, str] | None = None):
        self.objects = dict(objects or {"冰块": "固态", "水壶": "液态"})
        self.events: list[dict] = []

    def add_object(self, name: str, state: str) -> None:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        self.objects[name] = state
        self.events.append({"type": "new_object", "obj": name,
                            "state": state})

    def act(self, obj: str, action: str) -> dict:
        """Apply an action; returns {before, event, after} or an explicit
        refusal for unknown objects/actions (never a guessed outcome)."""
        if obj not in self.objects:
            return {"ok": False, "error": f"unknown object {obj!r}"}
        if action not in ACTIONS:
            return {"ok": False, "error": f"unknown action {action!r}; "
                                          f"known: {ACTIONS}"}
        before = self.objects[obj]
        after = DYNAMICS[(before, action)]
        self.objects[obj] = after
        ev = {"type": "transition", "obj": obj, "action": action,
              "before": before, "after": after}
        self.events.append(ev)
        return {"ok": True, **ev}

    def examples(self, actions=(("heat",), ("cool",)), repeats: int = 3
                 ) -> list[dict]:
        """All (state, action) -> new-state pairs as learnable examples.

        repeats: independent observations of the SAME transition with
        distinct evidence ids (morning/evening/...). Independent repeats
        are separate evidence (same rule as corpus corroboration), so
        train/heldout splits must use different repeats, never copies.
        """
        rows = []
        for state in STATES:
            for (a,) in actions:
                for i in range(repeats):
                    rows.append({
                        "id": f"{state}-{a}-obs{i}",
                        "inputs": {"state": state, "action": a},
                        "outputs": {"label": DYNAMICS[(state, a)]},
                        "evidence": f"obs-{i}"})
        return rows

    def scripted_day(self) -> list[dict]:
        """A fixed, repeatable episode used by tests (not hidden)."""
        plan = [("冰块", "heat"), ("冰块", "heat"), ("水壶", "cool"),
                ("水壶", "heat")]
        return [self.act(o, a) for o, a in plan]
