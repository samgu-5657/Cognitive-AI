"""World-instance / state-snapshot / simulation-event persistence.

Service layer only: every SQLite access goes through public
``memory.Memory`` APIs, every schema upgrade through
``schema_evolver.SchemaEvolver``. No direct connections, no
knowledge-table migration, no writes of simulation results into the
legacy knowledge stores (attributes / relations / claims / rules are
never touched here).

Only ``applied`` M4-C transition results are persisted, atomically
(state row + fact rows + event row in one transaction). Anything else
is refused before any write happens.
"""
from __future__ import annotations

import dataclasses
import json
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional

from .memory import Memory
from .models import (WorldEvent, WorldState, WorldStateFact,
                     WorldTransitionResult)

STATE_FORMAT = "world-state-v1"
EVENT_FORMAT = "world-event-v1"

_TYPE_REGISTRY: dict = {}


def _register(cls):
    _TYPE_REGISTRY[cls.__name__] = cls
    return cls


def _register_model_types() -> None:
    if _TYPE_REGISTRY:
        return
    from . import models as _m
    for name in ("WorldObjectRef", "WorldTerm", "WorldProposition",
                 "WorldStateFact", "WorldState", "WorldPatternTerm",
                 "WorldLiteralPattern", "WorldEffect", "WorldActionSchema",
                 "WorldPreconditionCheck", "WorldEvent",
                 "WorldTransitionResult", "WorldRuleView", "WorldClaimView",
                 "WorldEvidenceRef", "WorldFactView", "WorldSnapshot"):
        _register(getattr(_m, name))


def to_jsonable(obj: Any) -> Any:
    """Convert frozen world-model objects to deterministic JSON data.

    dicts (incl. MappingProxyType) recurse, tuples/lists become lists,
    Enums become their values, frozen dataclasses become tagged dicts.
    Anything else raises TypeError -- never str(obj) fallback.
    """
    _register_model_types()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, MappingProxyType):
        obj = dict(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"non-string mapping key: {k!r}")
            out[k] = to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj):
        data = {"__type__": type(obj).__name__}
        for f in dataclasses.fields(obj):
            data[f.name] = to_jsonable(getattr(obj, f.name))
        return data
    raise TypeError(f"unserializable object: {type(obj).__name__}")


def from_jsonable(data: Any) -> Any:
    """Rebuild world-model objects; unknown tags raise ValueError."""
    _register_model_types()
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return [from_jsonable(v) for v in data]
    if isinstance(data, dict):
        tag = data.get("__type__")
        if tag is None:
            return {k: from_jsonable(v) for k, v in data.items()}
        cls = _TYPE_REGISTRY.get(tag)
        if cls is None:
            raise ValueError(f"unknown serialized type: {tag!r}")
        kwargs = {k: from_jsonable(v)
                  for k, v in data.items() if k != "__type__"}
        try:
            return cls(**kwargs)
        except TypeError as e:
            raise ValueError(f"cannot rebuild {tag}: {e}")
    raise TypeError(f"unserializable data: {type(data).__name__}")


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def encode_state(state: WorldState) -> str:
    return _dumps({"format": STATE_FORMAT,
                   "state": to_jsonable(state)})


def decode_state(text: str) -> WorldState:
    try:
        payload = json.loads(text)
    except ValueError as e:
        raise ValueError(f"state payload is not JSON: {e}")
    if not isinstance(payload, dict) or \
            payload.get("format") != STATE_FORMAT:
        raise ValueError("unknown state format (no auto-upgrade)")
    state = from_jsonable(payload.get("state"))
    if not isinstance(state, WorldState):
        raise ValueError("state payload did not decode to WorldState")
    return state


def encode_event(event: WorldEvent) -> str:
    return _dumps({"format": EVENT_FORMAT,
                   "event": to_jsonable(event)})


def decode_event(text: str) -> WorldEvent:
    try:
        payload = json.loads(text)
    except ValueError as e:
        raise ValueError(f"event payload is not JSON: {e}")
    if not isinstance(payload, dict) or \
            payload.get("format") != EVENT_FORMAT:
        raise ValueError("unknown event format (no auto-upgrade)")
    event = from_jsonable(payload.get("event"))
    if not isinstance(event, WorldEvent):
        raise ValueError("event payload did not decode to WorldEvent")
    return event


def encode_fact(fact: WorldStateFact) -> str:
    return _dumps(to_jsonable(fact))


def decode_fact(text: str) -> WorldStateFact:
    try:
        payload = json.loads(text)
    except ValueError as e:
        raise ValueError(f"fact payload is not JSON: {e}")
    fact = from_jsonable(payload)
    if not isinstance(fact, WorldStateFact):
        raise ValueError("fact payload did not decode to WorldStateFact")
    return fact


class WorldPersistence:
    """Persist world instances, snapshots and simulation events."""

    def __init__(self, mem: Memory):
        self.mem = mem

    def ensure_storage(self) -> dict:
        from .schema_evolver import SchemaEvolver
        return SchemaEvolver(self.mem).ensure_world_model_storage()

    # ---- worlds ----
    def create_world(self, world_id: str, name: str,
                     initial_state: WorldState) -> dict:
        if not world_id or not isinstance(world_id, str):
            return {"ok": False, "error": "world_id must be non-empty"}
        if not isinstance(initial_state, WorldState):
            return {"ok": False, "error": "initial_state must be WorldState"}
        if initial_state.parent_state_id:
            return {"ok": False,
                    "error": "initial state must not have a parent"}
        if initial_state.step_index != 0:
            return {"ok": False,
                    "error": "initial state step_index must be 0"}
        if self.mem.get_world(world_id) is not None:
            return {"ok": False,
                    "error": f"world already exists: {world_id}"}
        if self.mem.get_world_state(initial_state.state_id) is not None:
            return {"ok": False, "error": "state id already used: "
                    f"{initial_state.state_id}"}
        from .memory import WORLD_MODEL_STORAGE_VERSION
        ops = [(
            "INSERT INTO worlds(world_id,name,storage_format_version)"
            " VALUES(?,?,?)",
            (world_id, name or "", WORLD_MODEL_STORAGE_VERSION),
        ), (
            "INSERT INTO world_states(state_id,world_id,parent_state_id,"
            "step_index,state_format_version) VALUES(?,?,?,?,?)",
            (initial_state.state_id, world_id, None, 0, STATE_FORMAT),
        )]
        for i, fact in enumerate(initial_state.facts):
            ops.append((
                "INSERT INTO world_state_facts(state_id,fact_index,content)"
                " VALUES(?,?,?)",
                (initial_state.state_id, i, encode_fact(fact)),
            ))
        try:
            self.mem.transact(ops)
        except Exception as e:
            return {"ok": False, "error": f"create rolled back: {e}"}
        return {"ok": True, "world_id": world_id,
                "state_id": initial_state.state_id}

    def get_world(self, world_id: str) -> Optional[dict]:
        return self.mem.get_world(world_id)

    def list_worlds(self) -> list[dict]:
        return self.mem.list_worlds()

    # ---- states ----
    def get_state(self, state_id: str) -> Optional[WorldState]:
        row = self.mem.get_world_state(state_id)
        if row is None:
            return None
        facts = [decode_fact(r["content"])
                 for r in self.mem.get_world_fact_rows(state_id)]
        return WorldState(
            state_id=row["state_id"], step_index=row["step_index"],
            facts=tuple(facts),
            parent_state_id=row["parent_state_id"] or "")

    def get_initial_state(self, world_id: str) -> Optional[WorldState]:
        row = self.mem.get_initial_world_state(world_id)
        if row is None:
            return None
        return self.get_state(row["state_id"])

    def list_states(self, world_id: str) -> list[WorldState]:
        out = []
        for row in self.mem.list_world_states(world_id):
            state = self.get_state(row["state_id"])
            if state is not None:
                out.append(state)
        return out

    # ---- events ----
    def get_event(self, event_id: str) -> Optional[WorldEvent]:
        row = self.mem.get_world_event(event_id)
        if row is None:
            return None
        return decode_event(self._event_text(row))

    def list_events(self, world_id: str) -> list[WorldEvent]:
        return [e for e in
                (self.get_event(r["event_id"])
                 for r in self.mem.list_world_events(world_id))
                if e is not None]

    @staticmethod
    def _event_text(row: dict) -> str:
        return _dumps({"format": EVENT_FORMAT, "event": {
            "__type__": "WorldEvent",
            "event_id": row["event_id"],
            "action_id": row["action_id"],
            "bindings": json.loads(row["bindings"]),
            "from_state": row["pre_state_id"],
            "to_state": row["post_state_id"],
            "step_index": row["step_index"],
            "effects": json.loads(row["effects"]),
            "simulated": bool(row["simulated"]),
        }})

    # ---- history ----
    def history(self, state_id: str) -> dict:
        row = self.mem.get_world_state(state_id)
        if row is None:
            return {"ok": False,
                    "error": f"state not found: {state_id}"}
        world_id = row["world_id"]
        chain = []
        current: Optional[str] = state_id
        while current:
            cur = self.mem.get_world_state(current)
            if cur is None:
                return {"ok": False,
                        "error": f"history broken at {current}"}
            if cur["world_id"] != world_id:
                return {"ok": False,
                        "error": "cross-world parent link refused: "
                        f"{current}"}
            chain.append(current)
            current = cur["parent_state_id"]
        chain.reverse()
        states = [self.get_state(s) for s in chain]
        events = []
        for s in chain[1:]:
            ev = self.mem.get_world_event_by_post(s)
            if ev is None:
                return {"ok": False,
                        "error": f"no creation event for state {s}"}
            events.append(self.get_event(ev["event_id"]))
        return {"ok": True, "states": states, "events": events}

    # ---- transitions ----
    def save_transition(self, result: WorldTransitionResult) -> dict:
        if not isinstance(result, WorldTransitionResult):
            return {"ok": False, "error": "not a transition result"}
        if result.outcome != "applied":
            return {"ok": False,
                    "error": f"only applied results persist, "
                    f"got {result.outcome}"}
        new_state = result.new_state
        event = result.event
        if new_state is None or event is None:
            return {"ok": False,
                    "error": "applied result lacks state or event"}
        if event.from_state != new_state.parent_state_id or \
                event.to_state != new_state.state_id:
            return {"ok": False,
                    "error": "event/state id mismatch"}
        if not new_state.parent_state_id:
            return {"ok": False,
                    "error": "non-initial state needs a parent"}
        pre = self.mem.get_world_state(new_state.parent_state_id)
        if pre is None:
            return {"ok": False,
                    "error": "pre-state not persisted: "
                    f"{new_state.parent_state_id}"}
        world_id = pre["world_id"]
        if new_state.step_index != pre["step_index"] + 1:
            return {"ok": False, "error": "step index must be parent+1"}
        if event.step_index != new_state.step_index:
            return {"ok": False, "error": "event step must match state"}
        if event.simulated is not True:
            return {"ok": False, "error": "only simulated events persist"}
        if self.mem.get_world_state(new_state.state_id) is not None:
            return {"ok": False, "error": "state id already used: "
                    f"{new_state.state_id}"}
        if self.mem.get_world_event(event.event_id) is not None:
            return {"ok": False, "error": "event id already used: "
                    f"{event.event_id}"}
        ops = [(
            "INSERT INTO world_states(state_id,world_id,parent_state_id,"
            "step_index,state_format_version) VALUES(?,?,?,?,?)",
            (new_state.state_id, world_id, new_state.parent_state_id,
             new_state.step_index, STATE_FORMAT),
        )]
        for i, fact in enumerate(new_state.facts):
            ops.append((
                "INSERT INTO world_state_facts(state_id,fact_index,content)"
                " VALUES(?,?,?)",
                (new_state.state_id, i, encode_fact(fact)),
            ))
        ops.append((
            "INSERT INTO world_events(event_id,world_id,action_id,"
            "pre_state_id,post_state_id,step_index,bindings,effects,"
            "simulated) VALUES(?,?,?,?,?,?,?,?,?)",
            (event.event_id, world_id, event.action_id, event.from_state,
             event.to_state, event.step_index,
             _dumps(to_jsonable([list(b) for b in event.bindings])),
             _dumps(to_jsonable(list(event.effects))),
             1 if event.simulated else 0),
        ))
        try:
            self.mem.transact(ops)
        except Exception as e:
            return {"ok": False, "error": f"save rolled back: {e}"}
        return {"ok": True, "state_id": new_state.state_id,
                "event_id": event.event_id, "world_id": world_id}
