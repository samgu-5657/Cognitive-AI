"""Unified task representation (Phase 4C).

CognitiveTask turns a user question into an actionable structure instead
of raw text. Every task must answer: known facts? goal? usable knowledge?
operations? unclear items? how is correctness judged?
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field


@dataclass
class CognitiveTask:
    task_id: str = ""
    task_type: str = "unknown"  # math | conceptual | logic | unknown
    input_text: str = ""
    input_frame: dict = field(default_factory=dict)
    known_facts: list = field(default_factory=list)
    goal: dict = field(default_factory=dict)
    constraints: list = field(default_factory=list)
    required_operations: list = field(default_factory=list)
    unresolved_items: list = field(default_factory=list)
    status: str = "created"  # created|planned|running|verified|failed|gap
    answer: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    def is_actionable(self) -> bool:
        return (self.task_type != "unknown" and bool(self.goal)
                and not self.unresolved_items
                and self.status in ("created", "planned"))


def build_task(text: str, frame: dict, memory) -> CognitiveTask:
    """Route a semantic frame to a task type with goal/ops/gaps filled.

    Unparseable or ambiguous frames become non-actionable tasks with
    unresolved_items (clarification required, never forced inference).
    """
    from .cog_planners import plan_for_frame
    task = CognitiveTask(input_text=text, input_frame=frame)
    if frame:
        plan_for_frame(task, frame, memory)
    if task.task_type == "unknown":
        if not frame or frame.get("intent") in ("unknown", None):
            task.unresolved_items.append("无法可靠解析输入,需要澄清")
        elif not task.unresolved_items:
            task.unresolved_items.append("无法确定任务类型,需要澄清")
    if frame and frame.get("unresolved_references"):
        task.unresolved_items.append(
            f"未消解指代: {frame['unresolved_references']}")
    if frame and any("pronoun" in a for a in frame.get("ambiguities", [])):
        task.unresolved_items.append(
            f"指代歧义: {frame['ambiguities']}")
    return task
