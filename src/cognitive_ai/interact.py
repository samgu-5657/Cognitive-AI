"""Interactive teaching session (Phase 4A).

`Session` owns no I/O: `handle(line) -> list[str]` is pure logic over the
shared Memory, fully testable. The `chat` REPL in cli.py only does
print/input (UTF-8). Every mutating path reuses existing modules
(learner / lifecycle / queue / gate / loop); nothing here writes around
the evidence gates.

Operation types (never substituted for each other):
  observation    facts + source  -> learn_from_observation (hypothesis rows)
  training_label reviewed labels -> UnknownQueue.review ONLY (no KB writes)
  hypothesis     unverified idea -> Claim(status=hypothesis), no KB writes
  correction     user fix        -> contested records + experience w/ reason
  research_request               -> curiosity question or budgeted explore

Natural language (Phase 4B): 理解 <text> parses to a SemanticFrame
(candidate only), shows the planned operation, and waits for 确认.
Pronouns resolve against the per-session persisted context; ambiguous
or failed parses produce zero knowledge writes.

Unparseable input produces zero database writes.
"""
from __future__ import annotations

import json
import os
import uuid

from .curiosity import Curiosity
from .learner import Learner
from .memory import Memory
from .novelty import NoveltyDetector, parse_chinese_simple
from .ontology import Ontology
from .reasoner import Reasoner
from .semantics import (SemanticFrame, parse_sentence, resolve_frame,
                        update_context)

ALIASES = {
    "帮助": "help", "退出": "quit", "拜拜": "quit",
    "概念": "concepts", "查看": "inspect", "分类": "predict",
    "观察": "observe", "教": "teach", "标签": "label",
    "假设": "hypothesis", "纠正": "correct", "探索": "explore",
    "经历": "experiences", "历史": "history", "模型": "model",
    "队列": "queue", "审核": "review", "状态": "state",
    "取消": "cancel", "理解": "understand", "确认": "confirm",
    "上下文": "context", "清空上下文": "clear_context",
    "解题": "solve", "求解": "solve", "任务": "task",
    "学规则": "rule_learn", "词义": "lex",
    "对话": "dialog",
    "说法": "phrase", "说法归纳": "phrase_induce",
    "说法验证": "phrase_validate", "说法激活": "phrase_activate",
    "说法列表": "phrase_list",
    "语义示例": "sem", "语义归纳": "sem_induce",
    "语义验证": "sem_validate", "语义激活": "sem_activate",
    "语义列表": "sem_list",
    "构式": "construct", "构式归纳": "construct_induce",
    "构式验证": "construct_validate", "构式激活": "construct_activate",
    "构式列表": "construct_list",
}

TEACH_STAGES = ("attrs", "function", "category", "done")


class Session:
    def __init__(self, memory: Memory, corpus_dir: str,
                 models_root: str | None = None,
                 queue_path: str | None = None):
        self.mem = memory
        self.learner = Learner(memory, corpus_dir)
        self.onto = Ontology(memory)
        self.reasoner = Reasoner(memory)
        self.detector = NoveltyDetector(memory)
        self.curiosity = Curiosity(memory)
        self.models_root = models_root
        self.queue_path = queue_path
        self.pending_teach: dict | None = None
        self.pending_rule: dict | None = None
        self.pending_frame: dict | None = None
        self.pending_define: dict | None = None
        self.pending_clarify: dict | None = None
        self.last_answer: dict | None = None
        self.session_id = uuid.uuid4().hex[:12]
        self._adapter = None
        self._ood_bundle = "unloaded"

    # ---- neural (lazy, optional) ----
    def adapter(self):
        if self._adapter is False:
            return None
        if self._adapter is not None:
            return self._adapter
        try:
            from .neural.adapter import NeuralAdapter
            from .neural.gate import read_active
            tag = (read_active(self.models_root) or {}).get("tag", "cnn-v1") \
                if self.models_root else "cnn-v1"
            root = os.path.join(self.models_root or "models", tag)
            self._adapter = NeuralAdapter(os.path.join(root, "best.pt"),
                                          os.path.join(root, "vocab.json"))
        except Exception:
            self._adapter = False
            return None
        return self._adapter

    def ood_bundle(self):
        if self._ood_bundle != "unloaded":
            return self._ood_bundle
        self._ood_bundle = None
        try:
            if self.models_root:
                from .neural.gate import read_active
                tag = (read_active(self.models_root) or {}).get(
                    "tag", "cnn-v1")
                from .neural.ood import load_artifacts
                arts = load_artifacts(
                    os.path.join(self.models_root, tag))
                if arts["prototypes.json"] and arts["ood_thresholds.json"]:
                    self._ood_bundle = {
                        "protos": arts["prototypes.json"],
                        "thresholds": arts["ood_thresholds.json"],
                        "temperature": 1.0}
        except Exception:
            self._ood_bundle = None
        return self._ood_bundle

    # ---- entry ----
    def _lexicon(self) -> dict:
        try:
            return {r["word"]: r["key"]
                    for r in self.mem.list_mappings()}
        except Exception:
            return {}

    def _phrase_rules(self) -> list[dict]:
        """Active learned rewrite rules for the parser (Phase 6C).

        Empty until the user teaches phrasings and explicitly activates
        a validated rule; the parser uses them as fallback only."""
        try:
            return self.mem.list_phrase_rules(status="active")
        except Exception:
            return []

    def _sem_rules(self) -> list[dict]:
        """Active direct-frame rules (Phase 6E).

        Built by semlearn.induce from labeled examples, validated on
        held-out entities, explicitly activated. dialog.process tries
        them only when the fixed parser returns unknown."""
        try:
            return self.mem.list_sem_rules(status="active")
        except Exception:
            return []

    def _constructions(self) -> list[dict]:
        """Active bidirectional constructions (Phase 6H-2B).

        Parse and generate read the same rows; dialog tries parsing
        only when the fixed parser (plus older learners) is unknown.
        """
        try:
            return self.mem.list_constructions(status="active")
        except Exception:
            return []

    def handle(self, line: str) -> list[str]:
        line = (line or "").strip()
        parts = line.split()
        if not parts:
            # empty line advances staged flows (skip), else ignored
            if self.pending_teach:
                return self._teach_answer(line)
            if self.pending_rule:
                return self._rule_next(self.pending_rule)
            return ["(空输入,已忽略)"]
        parts = line.split()
        cmd = ALIASES.get(parts[0], parts[0]).lower()
        args = parts[1:]
        if self.pending_teach and cmd not in ("cancel", "quit"):
            return self._teach_answer(line)
        if self.pending_rule and cmd not in ("cancel", "quit"):
            return self._rule_answer(line)
        fn = {"help": self._help, "quit": self._quit,
              "concepts": self._concepts, "inspect": self._inspect,
              "predict": self._predict, "observe": self._observe,
              "teach": self._teach, "label": self._label,
              "hypothesis": self._hypothesis, "correct": self._correct,
              "explore": self._explore, "experiences": self._experiences,
              "history": self._history, "model": self._model,
              "queue": self._queue, "review": self._review,
              "state": self._state, "cancel": self._cancel,
              "understand": self._understand, "confirm": self._confirm,
              "context": self._context, "clear_context": self._clear_context,
              "solve": self._solve, "task": self._task,
              "rule_learn": self._rule_learn, "lex": self._lex,
              "dialog": self._dialog,
              "phrase": self._phrase, "phrase_induce": self._phrase_induce,
              "phrase_validate": self._phrase_validate,
              "phrase_activate": self._phrase_activate,
              "phrase_list": self._phrase_list,
              "sem": self._sem, "sem_induce": self._sem_induce,
              "sem_validate": self._sem_validate,
              "sem_activate": self._sem_activate,
              "sem_list": self._sem_list,
              "construct": self._construct,
              "construct_induce": self._construct_induce,
              "construct_validate": self._construct_validate,
              "construct_activate": self._construct_activate,
              "construct_list": self._construct_list,
              }.get(cmd)
        if fn is None:
            return self._fallback(line)
        try:
            return fn(args)
        except Exception as e:
            return [f"执行出错(未写入可疑数据): {e}"]

    # ---- help / quit ----
    def _help(self, args) -> list[str]:
        return [
            "可用命令 (中文快捷等效):",
            "  概念/concepts                 列出概念",
            "  查看/inspect <id>             概念详情",
            "  分类/predict <文本>           模型分类+未知判断+规则解释",
            "  教/teach <名称>               分步教新事物(可随时取消)",
            "  观察/observe <名称> k=v ...   直接记录结构化观察 [observation]",
            "  标签/label <样本id> <L1> [L2] 审核训练标签 [training_label]",
            "  假设/hypothesis <文本>        存假设,不写知识 [hypothesis]",
            "  纠正/correct <名称> <属性> <新值> [原因]  纠错留痕 [correction]",
            "  探索/explore [问题] [--steps N]  请求探索 [research_request]",
            "  状态/state [名称]             显示认知状态与证据",
            "  理解/understand <句子>        解析自然语言,显示将执行的操作",
            "  确认/confirm                  执行待确认的解析(无待确认则空转)",
            "  解题/solve <问题>             统一认知执行(数学/概念/逻辑)",
            "  任务/task <id>                查看已保存任务的推理轨迹",
            "  学规则/rule_learn <名称> <领域>  分步提交示例并归纳候选规则",
            "  词义/lex <词语> <标准键>      教一个属性词映射(如 把手 has_handle)",
            "  对话/dialog <文本>           统一对话入口(寒暄/问答/追问/教学预览)",
            "  说法 <新说法> => <标准说法> [--留存]  教一种新表达(存示例,不改解析)",
            "  说法归纳/验证/激活/列表  从示例归纳改写模板并验证后显式激活",
            "  语义示例 <原句> => kind=<k>;entity=<e>;key=<k> [--留存]",
            "    直学语义帧(只收用途/材质/存在性三类查询,不改写句子)",
            "  语义归纳/验证/激活/列表  从标注示例归纳槽结构并验证后激活",
            "  构式 <原句> => entity=<实体>;value=<功能> [--留存]",
            "    教双向构式(解析与生成共用一条定义,需验证激活)",
            "  构式归纳/验证/激活/列表  从示例归纳双槽构式并验证后激活",
            "  上下文/context  清空上下文/clear_context  会话指代记忆管理",
            "  队列/queue [pending]  审核/review <id> -- 训练样本审核",
            "  经历/历史/模型  experiences/history/model",
            "  退出/quit  安全退出(数据已持久化)",
        ]

    def _quit(self, args) -> list[str]:
        self.pending_teach = None
        self.pending_rule = None
        return ["__quit__", "再见,所有已保存数据都在数据库中。"]

    def _cancel(self, args) -> list[str]:
        self.pending_teach = None
        self.pending_rule = None
        return ["已取消当前教学,已记录的内容保留,未填的保持未知。"]

    # ---- read paths ----
    def _concepts(self, args) -> list[str]:
        out = [f"{c['id']} [{c['kind']}] parents={self.mem.get_parents(c['id'])} "
               f"name={c['name']}" for c in self.onto.list()]
        return out or ["(空)"]

    def _inspect(self, args) -> list[str]:
        if not args:
            return ["用法: 查看 <概念id>"]
        try:
            d = self.onto.describe(args[0])
        except Exception as e:
            return [f"查无此概念: {e}"]
        lines = [f"{d['id']} name={d['name']} kind={d['kind']}",
                 f"  parents={d['parents']} children={d['children']}"]
        for a in d["attributes"]:
            lines.append(f"  attr {a['key']}={a['value']} "
                         f"[{a['status']}/{a['source']}]")
        for r in d["relations"]:
            lines.append(f"  rel -{r['rel_type']}-> {r['dst_id']} "
                         f"[{r['status']}]")
        return lines

    def _experiences(self, args) -> list[str]:
        n = int(args[0]) if args and args[0].isdigit() else 10
        return [f"#{e['id']} [{e['kind']}] {e['summary']}"
                for e in self.mem.list_experiences(limit=n)]

    def _history(self, args) -> list[str]:
        from .schema_evolver import SchemaEvolver
        ev = SchemaEvolver(self.mem)
        out = ["== versions =="]
        out += [f"  v{v['version']}: {v['reason']}" for v in ev.versions()]
        out += ["== changes =="]
        out += [f"  #{c['id']} {c['kind']} {c['status']} "
                f"v{c['version_before']}->v{c['version_after']} :: {c['reason']}"
                for c in ev.history()]
        return out

    def _model(self, args) -> list[str]:
        lines = []
        ad = self.adapter()
        if ad is None:
            lines.append("神经网络模型不可用(未安装或无 checkpoint)。")
        else:
            lines.append(f"活动模型: {ad.version}")
            if self.models_root:
                try:
                    from .neural.gate import read_active
                    lines.append(f"active 指针: {read_active(self.models_root)}")
                except Exception:
                    pass
        from .schema_evolver import SchemaEvolver
        lines.append(f"知识版本: v{SchemaEvolver(self.mem).current_version()}")
        return lines

    def _queue(self, args) -> list[str]:
        if not self.queue_path:
            return ["队列未配置。"]
        from .neural.queue import UnknownQueue
        rows = UnknownQueue(self.queue_path).list(
            status=args[0] if args else None)
        out = [f"[{r['sample_id']}] {r['text']!r} conf={r['confidence']} "
               f"review={r['review']['status']} label={r['label']}"
               for r in rows]
        return out or ["(队列为空)"]

    def _review(self, args) -> list[str]:
        # 审核 <id> <L1> [L2]  /  审核 <id> 拒绝
        if len(args) < 1:
            return ["用法: 审核 <样本id> <L1> [L2]  或  审核 <样本id> 拒绝"]
        if not self.queue_path:
            return ["队列未配置。"]
        from .neural.queue import UnknownQueue
        q = UnknownQueue(self.queue_path)
        sid = args[0]
        if len(args) == 2 and args[1] in ("拒绝", "reject"):
            return [str(q.review(sid, accept=False))]
        l1 = args[1] if len(args) > 1 else None
        l2 = args[2] if len(args) > 2 else None
        out = q.review(sid, accept=True, l1=l1, l2=l2, reviewer="chat")
        self.mem.add_experience("evaluation", f"chat review {sid}: {out}",
                                {"op": "training_label", **out})
        return [str(out)]

    # ---- predict + state (explanations from real structures) ----
    def _predict(self, args) -> list[str]:
        if not args:
            return ["用法: 分类 <文本>"]
        return self._explain(" ".join(args))

    def _state(self, args) -> list[str]:
        if not args:
            return ["用法: 状态 <名称>"]
        return self._explain(args[0], as_state=True)

    def _explain(self, text: str, as_state: bool = False) -> list[str]:
        lines = [f"输入: {text}"]
        # 1. rule parse
        parsed = parse_chinese_simple(text)
        lines.append(f"规则解析: {parsed}")
        # 2. KB match
        matches = self.mem.find_concepts_by_name(text)
        lines.append(f"概念匹配: {[c['id'] for c in matches] or '无(未知名称)'}")
        # 3. neural candidate + OOD
        ad = self.adapter()
        if ad is None:
            lines.append("模型分类: 不可用")
        else:
            try:
                cand = ad.predict(text)
                lines.append(f"模型分类: {cand['candidate_category']} "
                             f"conf={cand['confidence']} "
                             f"verdict={cand['verdict']} [{cand['model_version']}]")
            except Exception as e:
                lines.append(f"模型分类失败: {e}")
                cand = None
            bundle = self.ood_bundle()
            if bundle:
                try:
                    from .neural.ood import assess
                    second = None
                    v = assess(text, ad.model, ad.vocab, bundle["protos"],
                               bundle["temperature"], bundle["thresholds"],
                               second=second)
                    lines.append(f"未知判断: {v['verdict']} "
                                 f"({' ; '.join(v.get('reasons', []))})")
                except Exception as e:
                    lines.append(f"未知判断失败: {e}")
        # 4. rule classify verdict
        cls = self.learner.classify({"name": text, "attributes": {}})
        lines.append(f"规则归类: best={cls['best']} verdict={cls['verdict']}")
        # 5. related knowledge for matches
        for c in matches[:2]:
            d = self.onto.describe(c["id"])
            lines.append(f"已知[{c['id']}]: parents={d['parents']} "
                         f"attrs={[(a['key'], a['value'], a['status']) for a in d['attributes']]}")
        # 6. gaps
        gaps = [g for g in self.detector.knowledge_gaps()
                if not matches or g.get("concept") in [c["id"] for c in matches]]
        if gaps:
            lines.append(f"知识缺口(相关): {gaps[:5]}")
        else:
            lines.append("知识缺口(相关): 无")
        lines.append("结论: 以上均来自实时查询,未做猜测;未知项保持未知。")
        return lines

    # ---- mutating ops, one type each ----
    def _observe(self, args) -> list[str]:
        # 观察 <名称> k=v ...   [observation]
        if not args:
            return ["用法: 观察 <名称> [属性=值 ...]"]
        name, attrs = args[0], {}
        for tok in args[1:]:
            if "=" in tok or "：" in tok:
                k, v = tok.replace("：", "=", 1).split("=", 1)
                attrs[k.strip()] = v.strip()
        res = self.learner.learn_from_observation(
            {"name": name, "attributes": attrs, "source": "chat"})
        nov = res.get("novelty", {})
        lines = [f"[observation] 已记录观察 '{name}' (经历#{res.get('experience_id')})"]
        lines.append(f"匹配概念: {nov.get('matched') or '无(未知名称)'}")
        if nov.get("needs_new_concept"):
            lines.append("新名称: 已建暂定概念,仍缺区分证据,保持未知部分。")
        if nov.get("conflicts"):
            lines.append(f"冲突(已保留双方,未覆盖): {nov['conflicts']}")
        if res.get("new_concept", {}).get("id"):
            lines.append(f"暂定概念: {res['new_concept']['id']}")
        return lines

    def _teach(self, args) -> list[str]:
        # 教 <名称>  -> staged observation flow
        if not args:
            return ["用法: 教 <名称>  (随后按提示补充,不必一次答全)"]
        name = args[0]
        matches = self.mem.find_concepts_by_name(name)
        self.pending_teach = {"name": name, "stage": "attrs",
                              "known": bool(matches)}
        from .semantics import SemanticFrame as _F
        self.mem.set_context(
            self.session_id,
            update_context(self.mem.get_context(self.session_id),
                           _F(intent="observe",
                              entities=[{"text": name, "role": "object"}])))
        lines = [f"[teach] 开始记录 '{name}'。"]
        lines.append(f"已有概念: {[c['id'] for c in matches] or '无,这是新事物'}")
        lines.append("请提供可观察的属性,形如: 材质=陶瓷 颜色=白 (直接回车跳过,未知保持未知)")
        return lines

    def _teach_answer(self, line: str) -> list[str]:
        st = self.pending_teach
        assert st is not None
        stage = st["stage"]
        if stage == "attrs":
            attrs = {}
            for tok in line.split():
                if "=" in tok or "：" in tok:
                    k, v = tok.replace("：", "=", 1).split("=", 1)
                    if k.strip():
                        attrs[k.strip()] = v.strip()
            res = self.learner.learn_from_observation(
                {"name": st["name"], "attributes": attrs, "source": "chat"})
            st["stage"] = "function"
            return [f"[observation] 属性已记 {attrs or '(跳过)'}",
                    "它的功能/用途是什么? (如: 喝水 / 直接回车跳过)"]
        if stage == "function":
            if line.strip():
                self.learner.learn_from_observation(
                    {"name": st["name"], "attributes": {"function": line.strip()},
                     "source": "chat"})
                lines = [f"[observation] 功能已记: {line.strip()}"]
            else:
                lines = ["[observation] 功能跳过,保持未知。"]
            st["stage"] = "category"
            return lines + ["它属于哪类? (如: 容器 / 直接回车跳过)"]
        if stage == "category":
            want = line.strip()
            pid = None
            if want:
                if self.mem.get_concept(want) is not None:
                    pid = want
                else:
                    hits = self.mem.find_concepts_by_name(want)
                    pid = hits[0]["id"] if hits else None
            if want and pid is None:
                # unknown parent: do NOT invent hierarchy; keep pending info
                self.mem.add_experience(
                    "observation", f"teach '{st['name']}' 提到未知类别 '{want}'",
                    {"op": "observation", "pending_parent": want})
                out = [f"[observation] 类别'{want}'暂无对应概念,已记为待确认,不虚构归属。"]
            elif want:
                self.mem.add_experience(
                    "observation", f"teach '{st['name']}' 自述属于 {pid}",
                    {"op": "observation", "claimed_parent": pid})
                out = [f"[observation] 已记自述类别 {pid} (待证据确认,未改结构)。"]
            else:
                out = ["[observation] 类别跳过,保持未知。"]
            st["stage"] = "done"
            self.pending_teach = None
            return out + ["教学暂告段落。可用'状态 <名称>'查看,未知项仍可后续补充。"]
        self.pending_teach = None
        return ["教学状态异常,已重置。"]

    def _label(self, args) -> list[str]:
        # 标签 <样本id> <L1> [L2]  [training_label: review only]
        if len(args) < 2:
            return ["用法: 标签 <样本id> <L1> [L2]  (只做审核,不写知识)"]
        if not self.queue_path:
            return ["队列未配置。"]
        from .neural.queue import UnknownQueue
        out = UnknownQueue(self.queue_path).review(
            args[0], accept=True, l1=args[1],
            l2=args[2] if len(args) > 2 else None, reviewer="chat")
        self.mem.add_experience("evaluation", f"chat label {args[0]}: {out}",
                                {"op": "training_label", **out})
        return [f"[training_label] {out}",
                "注意: 这只是审核标签,要训练模型请走 neural-incremental + 门禁。"]

    def _hypothesis(self, args) -> list[str]:
        # 假设 <文本>  [hypothesis: claim only]
        if not args:
            return ["用法: 假设 <文本>"]
        text = " ".join(args)
        cid = self.mem.add_claim(text, subject_id=None,
                                 status="hypothesis", confidence=0.4)
        self.mem.add_experience("observation", f"chat hypothesis #{cid}",
                                {"op": "hypothesis", "claim_id": cid})
        return [f"[hypothesis] 已存假设#{cid},未写入知识,待证据。"]

    def _correct(self, args) -> list[str]:
        # 纠正 <名称> <属性> <新值> [原因]  [correction]
        if len(args) < 3:
            return ["用法: 纠正 <名称> <属性> <新值> [原因]"]
        name, key, new = args[0], args[1], args[2]
        reason = " ".join(args[3:]) or "用户纠正"
        res = self.learner.learn_from_observation(
            {"name": name, "attributes": {key: new}, "source": "chat-correction"})
        self.mem.add_experience(
            "evaluation", f"chat correction: {name}.{key} -> {new}",
            {"op": "correction", "reason": reason, "result": str(res)[:500]})
        return [f"[correction] 已记录纠正并保留原值(原因: {reason})。",
                "旧值未被静默覆盖,冲突双方可在'状态'中查看。"]

    def _explore(self, args) -> list[str]:
        # 探索 [问题] [--steps N]  [research_request, budgeted]
        steps, words = 3, []
        for i, a in enumerate(args):
            if a == "--steps" and i + 1 < len(args) and args[i + 1].isdigit():
                steps = min(int(args[i + 1]), 20)
        words = [a for a in args if a != "--steps" and not a.isdigit()]
        if words:
            qid = self.mem.add_question(" ".join(words), "gap", None, 0.5,
                                        "queued", {"source": "chat"})
            self.mem.add_experience("evaluation", "chat research request",
                                    {"op": "research_request",
                                     "question_id": qid})
            return [f"[research_request] 已加入研究队列#{qid}。"]
        from .loop import ExplorationLoop
        import os as _os
        corp = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             "..", "..", "data", "corpus")
        out = ExplorationLoop(self.mem, corp).run(steps=steps,
                                                  max_seconds=60.0)
        lines = [f"[research_request] 探索 {out['steps_executed']} 步，用时{out['elapsed_s']}s"]
        for e in out["log"]:
            lines.append(f"  step{e.get('step')}: Q{e.get('question_id')} "
                         f"{e.get('result')} action={e.get('action')}")
        return lines

    # ---- natural language understanding (Phase 4B) ----
    def _understand(self, args) -> list[str]:
        # 理解 <句子>: parse -> resolve -> preview op, wait for 确认
        if not args:
            return ["用法: 理解 <中文句子>  (解析后用'确认'执行)"]
        text = " ".join(args)
        frame = parse_sentence(text, source="user",
                               lexicon=self._lexicon(),
                               phrase_rules=self._phrase_rules())
        state = self.mem.get_context(self.session_id)
        frame = resolve_frame(frame, state)
        if frame.intent == "unknown":
            return ["暂时无法理解这句话(只支持帮助中列出的命令与常见陈述句式)。",
                    f"解析痕迹: {frame.trace}",
                    "请换一种表达,或用'观察 <名称> 属性=值'结构化输入。",
                    "本次未写入任何知识,未修改模型。"]
        self.mem.set_context(self.session_id, update_context(state, frame))
        self.pending_frame = frame.to_dict()
        lines = [f"原文: {text}",
                 f"意图: {frame.intent}  模态: {frame.modality} "
                 f"范围: {frame.scope}  置信度: {frame.confidence}",
                 f"实体: {frame.entities}",
                 f"谓词: {frame.predicates}"]
        if frame.unresolved_references:
            lines.append(f"未消解指代: {frame.unresolved_references}")
        if frame.ambiguities:
            lines.append(f"歧义/缺失: {frame.ambiguities}")
        lines.append(f"来源: {frame.source}")
        lines.append(f"将执行: {self._planned_op(frame)} "
                     "(输入'确认'执行,'取消'放弃;确认前不写入)")
        return lines

    @staticmethod
    def _planned_op(frame: SemanticFrame) -> str:
        return {"classify": "observation(暂定概念+自述归属待证)",
                "attribute": "observation(新属性记 hypothesis)",
                "function": "observation(功能记 hypothesis)",
                "negate": "denial 记录(Claim,不写属性)",
                "hedge": "hypothesis(Claim,待证据)",
                "correct": "correction(关联原判断,旧值保留)",
                "causal": "hypothesis(条件句,不当事实)",
                "observe": "observation(新实例暂定概念)",
                "query": "只读解释(不写库)",
                "unknown": "拒绝(不写库)"}.get(frame.intent, "拒绝(不写库)")

    def _confirm(self, args) -> list[str]:
        if not self.pending_frame:
            return ["当前没有待确认的解析。先用'理解 <句子>'。"]
        frame = self.pending_frame
        self.pending_frame = None
        # never execute while references are unresolved: it would create
        # literal 它-concepts out of pronouns
        if frame.get("unresolved_references"):
            return [f"指代未消解{frame['unresolved_references']},拒绝执行(无写入)。",
                    "请先说清对象,或用'观察 <名称> ...'直接指定。"]
        if any("pronoun" in a for a in frame.get("ambiguities", [])):
            return ["存在未澄清的指代歧义,拒绝执行(无写入)。",
                    f"歧义: {frame['ambiguities']}",
                    "请指明对象后重试。"]
        return self._execute_frame(frame)

    def _execute_frame(self, fr: dict) -> list[str]:
        intent, modality = fr["intent"], fr["modality"]
        if intent == "query":
            ent = (fr["entities"] + [{}])[0].get("text", "")
            return self._explain(ent or fr["raw"]) if ent else \
                ["问题缺少可查询的实体。"]
        if intent == "unknown":
            return ["该解析不可执行,已放弃(无写入)。"]
        if intent == "hedge" or modality == "possible":
            preds = fr["predicates"]
            text = f"{preds[0]}" if preds else fr["raw"]
            cid = self.mem.add_claim(f"可能: {text}", subject_id=None,
                                     status="hypothesis", confidence=0.4)
            self.mem.add_experience("observation", f"nlu hedge #{cid}",
                                    {"op": "hypothesis", "frame": fr})
            return [f"[hypothesis] 不确定陈述已存#{cid},未当事实(置信度{fr['confidence']})。"]
        if intent == "negate":
            preds = fr["predicates"]
            text = f"{preds[0]}" if preds else fr["raw"]
            cid = self.mem.add_claim(f"否定: {text}", subject_id=None,
                                     status="hypothesis", confidence=0.5)
            self.mem.add_experience("observation", f"nlu denial #{cid}",
                                    {"op": "hypothesis", "frame": fr,
                                     "note": "denial recorded, no attribute written"})
            return [f"[hypothesis] 否定已记录#{cid},未写入属性,未覆盖旧值。"]
        if intent == "correct":
            return self._execute_correction(fr)
        if intent == "causal":
            cid = self.mem.add_claim(f"条件: {fr['predicates']}",
                                     subject_id=None, status="hypothesis",
                                     confidence=0.4)
            self.mem.add_experience("observation", f"nlu causal #{cid}",
                                    {"op": "hypothesis", "frame": fr})
            return [f"[hypothesis] 条件句已存#{cid},不当事实,待验证。"]
        # classify / attribute / function / observe -> observation op
        return self._execute_observation(fr)

    def _execute_observation(self, fr: dict) -> list[str]:
        attrs: dict = {}
        parents: list[str] = []
        name = ""
        for p in fr["predicates"]:
            t = p.get("type")
            if t == "is_a" and not p.get("denied"):
                name = name or p.get("subj", "")
                obj = p.get("obj", "")
                hit = self.mem.get_concept(obj) or \
                    (self.mem.find_concepts_by_name(obj) or [None])[0]
                if hit:
                    parents.append(hit["id"])
                else:
                    self.mem.add_experience(
                        "observation", f"nlu claimed parent '{obj}' unknown",
                        {"op": "observation", "pending_parent": obj})
            elif t == "attr" and not p.get("denied") and p.get("key"):
                name = name or p.get("subj", "")
                attrs[p["key"]] = p.get("value")
            elif t == "capable" and not p.get("denied"):
                name = name or p.get("subj", "")
                attrs["function"] = p.get("action")
            elif t == "has" and not p.get("denied"):
                name = name or p.get("subj", "")
                attrs[p.get("feature", "feature")] = True
            elif t == "new_instance":
                name = name or p.get("obj", "")
        if not name and fr["entities"]:
            name = fr["entities"][0]["text"]
        if not name:
            return ["无法确定主体,已放弃(无写入)。"]
        res = self.learner.learn_from_observation(
            {"name": name, "parents": parents, "attributes": attrs,
             "source": "chat-nlu"})
        return [f"[observation] 已记录'{name}' attrs={attrs} "
                f"parents={parents or '未知(未虚构)'}",
                f"经历#{res.get('experience_id')} "
                f"novelty={str(res.get('novelty', {}))[:200]}"]

    def _execute_correction(self, fr: dict) -> list[str]:
        inner = (fr["predicates"] + [{}])[0].get("content") or {}
        # same guard as 确认: unresolved pronouns must not become concepts
        if inner.get("unresolved_references") or \
                any("pronoun" in a for a in inner.get("ambiguities", [])):
            self.mem.add_experience(
                "evaluation", "nlu correction blocked: unresolved reference",
                {"op": "correction", "frame": fr})
            return ["纠正内容指代未消解,拒绝执行(无写入)。请先说清对象。"]
        # antecedent: last session observation about the same entity
        state = self.mem.get_context(self.session_id)
        prior = None
        want = {e["text"] for e in fr.get("entities", [])}
        if state.get("last_observation"):
            lo = state["last_observation"]
            if want & set(lo.get("entities", [])):
                prior = lo
        lines = []
        if inner and inner.get("intent") not in ("unknown", None):
            lines += self._execute_frame({**inner, "source": "chat-correction"})
        self.mem.add_experience(
            "evaluation", f"nlu correction links prior={bool(prior)}",
            {"op": "correction", "frame": fr, "antecedent": prior})
        lines.append("纠正已关联原判断(见上);旧值保留,未静默覆盖。"
                     if prior else "未找到明确原判断,已按新观察记录,请用'经历'核对。")
        return lines

    def _context(self, args) -> list[str]:
        st = self.mem.get_context(self.session_id)
        return [f"会话 {self.session_id} turn={st['turn']} "
                f"entities={[e['text'] for e in st['entities']]}",
                f"last_observation={st.get('last_observation')}"]

    def _clear_context(self, args) -> list[str]:
        self.mem.clear_context(self.session_id)
        return ["当前会话上下文已清除(知识库不受影响)。"]

    def _solve(self, args) -> list[str]:
        # 解题 <问题>: 统一执行循环 + 基于轨迹的答案
        if not args:
            return ["用法: 解题 <数学/概念/逻辑问题>  (如: 解题 计算3+4*2)"]
        from .executor import solve_text
        from .explainer import explain
        from .semantics import parse_sentence as _parse
        text = " ".join(args)
        frame = _parse(text, source="user", lexicon=self._lexicon(),
                       phrase_rules=self._phrase_rules()).to_dict()
        trace = solve_text(text, frame, self.mem, self.onto, self.reasoner)
        ans = explain(trace, style="steps" if len(args) > 3 else "short")
        self.mem.add_experience(
            "evaluation", f"chat solve -> {trace['task']['status']}",
            {"task_id": trace["task"]["task_id"],
             "certainty": ans["certainty"]})
        return [f"[solve] {ans['text']}",
                f"任务{trace['task']['task_id']} "
                f"状态={trace['task']['status']} 确定度={ans['certainty']}"]

    def _task(self, args) -> list[str]:
        if not args:
            rows = self.mem.query(
                "SELECT task_id,task_type,status FROM cog_tasks ORDER BY "
                "rowid DESC LIMIT 10")
            return [f"{r['task_id']} [{r['task_type']}/{r['status']}]"
                    for r in rows] or ["(无任务记录)"]
        from .executor import load_trace
        tr = load_trace(self.mem, args[0])
        if not tr:
            return ["查无此任务。"]
        steps = tr.get("steps", [])
        out = [f"任务{args[0]} status={tr.get('task', {}).get('status')}"]
        for s in steps[:12]:
            out.append(f"  {s.get('op')}: {s.get('status')} "
                       f"{str(s.get('note', ''))[:80]}")
        return out

    # ---- rule teaching flow (Phase 4D): examples -> counterexamples ->
    # held-out -> induce (candidate only, never auto-promoted)
    def _rule_learn(self, args) -> list[str]:
        if len(args) < 2:
            return ["用法: 学规则 <名称> <领域math|concepts>  (随后分步给示例)"]
        name, domain = args[0], args[1]
        if domain not in ("math", "concepts"):
            return [f"领域仅支持 math/concepts,得到 {domain}。"]
        self.pending_rule = {"name": name, "domain": domain,
                             "stage": "examples", "examples": [],
                             "negatives": [], "heldout": []}
        return [f"[rule_learn] 为规则'{name}'({domain})收集示例。",
                "先给正例,每行一个,math形如 n=5 => 11；concepts形如 "
                "颜色=红,形状=圆 => 苹果。空行进入下一步。"]

    @staticmethod
    def _parse_rule_line(line: str, domain: str) -> dict | None:
        if "=>" not in line:
            return None
        left, right = line.split("=>", 1)
        if domain == "math":
            try:
                inputs, outputs = {}, {}
                for tok in left.split(","):
                    tok = tok.strip()
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        inputs[k.strip() or "n"] = v.strip()
                if "n" not in inputs and left.strip():
                    inputs = {"n": left.strip()}
                outputs = {"out": right.strip()}
                if not inputs or not outputs["out"]:
                    return None
                return {"inputs": inputs, "outputs": outputs}
            except Exception:
                return None
        inputs, outputs = {}, {}
        for tok in left.split(","):
            tok = tok.strip()
            if "=" in tok:
                k, v = tok.split("=", 1)
                inputs[k.strip()] = v.strip()
        outputs = {"label": right.strip()}
        return {"inputs": inputs, "outputs": outputs} if inputs and outputs["label"] else None

    def _rule_answer(self, line: str) -> list[str]:
        st = self.pending_rule
        assert st is not None
        if not line.strip():
            return self._rule_next(st)
        if st["stage"] == "examples":
            row = self._parse_rule_line(line, st["domain"])
            if row is None:
                return ["格式不对,形如 n=5 => 11 或 颜色=红 => 苹果；空行下一步。"]
            row["id"] = f"ex{len(st['examples'])}"
            st["examples"].append(row)
            return [f"正例+1(共{len(st['examples'])})。继续或空行下一步。"]
        if st["stage"] == "negatives":
            row = self._parse_rule_line(line, st["domain"])
            if row is None:
                return ["格式不对；空行下一步(可无反例)。"]
            row["id"] = f"neg{len(st['negatives'])}"
            st["negatives"].append(row)
            return [f"反例+1(共{len(st['negatives'])})。继续或空行下一步。"]
        if st["stage"] == "heldout":
            if line.strip() == "跳过":
                return self._rule_next(st, skip_heldout=True)
            row = self._parse_rule_line(line, st["domain"])
            if row is None:
                return ["格式不对；至少给一个独立验证例,或输入'跳过'。"]
            row["id"] = f"ho{len(st['heldout'])}"
            st["heldout"].append(row)
            return [f"验证例+1(共{len(st['heldout'])})。继续、空行诱导,或'跳过'。"]
        self.pending_rule = None
        return ["规则教学状态异常,已重置。"]

    def _rule_next(self, st: dict, skip_heldout: bool = False) -> list[str]:
        from . import rulelearn as RL
        if st["stage"] == "examples":
            if len(st["examples"]) < 2:
                return ["正例至少2个,否则无法归纳。继续给,或'取消'。"]
            st["stage"] = "negatives"
            return ["正例已够。现在给反例(应被拒绝的输入),每行一个；可直接空行跳过。"]
        if st["stage"] == "negatives":
            st["stage"] = "heldout"
            return ["反例阶段结束。现在给独立验证例(不能是上面的正例改写)；"
                    "至少1个,或输入'跳过'(跳过则只能存候选,不能验证)。"]
        # heldout -> induce
        self.pending_rule = None
        rep = RL.induce(st["examples"], st["negatives"], domain=st["domain"])
        lines = [f"[rule_learn] 归纳完成: {len(rep['candidates'])} 候选" +
                 ("(歧义:需反例或人工选择)" if rep["ambiguous"] else "")]
        for i, c in enumerate(rep["candidates"]):
            lines.append(f"  #{i} [{c['proposer']}] {c['form']}")
        if not rep["candidates"]:
            self.mem.add_experience(
                "evaluation", f"rule induce {st['name']}: no candidate",
                {"op": "rule_induce", "outcome": "rule"})
            return lines + ["无一致候选(示例矛盾或超出DSL),已记录,未建规则。"]
        rid = RL.new_rule_id(st["name"])
        self.mem.save_rule(RL.make_proposal_record(
            rid, st["name"], st["domain"], rep["candidates"],
            evidence=[{"source": "chat"}]))
        self.mem.add_experience(
            "evaluation", f"rule proposal {rid}",
            {"op": "rule_induce", "outcome": "rule",
             "candidates": len(rep["candidates"]),
             "ambiguous": rep["ambiguous"]})
        lines.append(f"已存候选提案 {rid}(状态candidate,未验证)。")
        if skip_heldout or not st["heldout"]:
            lines.append("无独立验证例:停在候选。用rule-validate验证后才能晋升。")
        else:
            lines.append(f"验证例{len(st['heldout'])}个已收:用 rule-validate "
                         f"{rid} --heldout <文件> 验证(示例需先导出成文件)。")
        if rep["ambiguous"]:
            lines.append(f"歧义未解:用 rule-select {rid} <编号> 选择后再验证。")
        return lines

    def _lex(self, args) -> list[str]:
        # 词义 <词语> <标准键>: learn one lexical mapping (recorded).
        if len(args) < 2:
            rows = self.mem.list_mappings()
            return ["已学映射:" + (", ".join(
                f"{r['word']}->{r['key']}" for r in rows) or "(无)"),
                    "用法: 词义 <词语> <标准键>"]
        word, key = args[0], args[1]
        self.mem.learn_mapping(word, key, evidence="chat-teach")
        self.mem.add_experience(
            "evaluation", f"chat lexicon: {word}->{key}",
            {"op": "lexical", "word": word, "key": key})
        return [f"[lexical] 已记住 {word}->{key},后续解析即用(经验已记)。"]

    def _dialog(self, args) -> list[str]:
        # 对话 <文本>: unified entry (smalltalk/QA/followup/teach preview).
        # Reuses dialog.process; never crashes, never writes on unknown.
        if not args:
            return ["用法: 对话 <文本>  (如: 对话 你好 / 对话 杯子有什么用途)"]
        from . import dialog as _dlg
        turn = _dlg.process(self, " ".join(args))
        ans = turn.get("answer") or {}
        lines = [f"[dialog:{turn.get('route')}] {ans.get('text', '')}"]
        if ans.get("certainty"):
            lines.append(f"确定度={ans['certainty']}")
        return lines

    # ---- phrase teaching flow (Phase 6C): examples -> induce (candidate)
    # -> held-out validate -> explicit activate. Learned rules only
    # reshape unknown inputs before the fixed parser; they never write
    # knowledge and never override a direct parse.
    def _phrase(self, args) -> list[str]:
        # 说法 <新说法> => <标准说法> [--留存]
        if "=>" not in " ".join(args):
            return ["用法: 说法 <新说法> => <标准说法> [--留存]  "
                    "(--留存: 记为独立验证例,不参与归纳)"]
        heldout = args[-1] == "--留存"
        body = " ".join(args[:-1] if heldout else args)
        src, dst = (p.strip() for p in body.split("=>", 1))
        from . import phraselearn as PL
        chk = PL.check_example(src, dst, self._lexicon())
        if not chk["ok"]:
            return [f"[phrase] 示例未收录: {chk['reason']}。"
                    "标准说法须能被现有解析器直接理解(可用'理解'先试)。"]
        eid = self.mem.add_phrase_example(
            src, dst, heldout, evidence="chat-teach")
        self.mem.add_experience(
            "evaluation", f"chat phrase example #{eid}",
            {"op": "phrase_teach", "heldout": heldout})
        kind = "验证例" if heldout else "训练例"
        return [f"[phrase] 已记{kind}#{eid}: {src!r} => {dst!r} "
                f"(目标意图={chk['signature']['intent']})。",
                "示例只存不改解析；用'说法归纳'从训练例归纳候选模板。"]

    def _phrase_induce(self, args) -> list[str]:
        from . import phraselearn as PL
        trains = self.mem.list_phrase_examples(heldout=False)
        if len(trains) < 1:
            return ["训练例不足(至少1个)。先用'说法 A => B'教学。"]
        rep = PL.induce(
            [{"src": r["src"], "dst": r["dst"]} for r in trains],
            self._lexicon())
        lines = [f"[phrase] 归纳完成: {len(rep['candidates'])} 候选, "
                 f"{len(rep['rejected'])} 组被拒(已说明原因,未建规则)。"]
        for c in rep["candidates"]:
            rid = PL.new_rule_id("chat")
            self.mem.save_phrase_rule(
                {"id": rid, "kind": c["kind"], "src_tmpl": c["src_tmpl"],
                 "dst_tmpl": c["dst_tmpl"], "status": "candidate",
                 "evidence": [{"source": "chat", "support": c["support"]}],
                 "report": {"note": c["note"],
                            "signature": c["signature"]}})
            lines.append(f"  {rid} [{c['kind']}] {c['note']} "
                         f"(状态candidate,未验证未激活)")
        for r in rep["rejected"]:
            lines.append(f"  拒: {r.get('reason')}")
        self.mem.add_experience(
            "evaluation", "chat phrase induce",
            {"op": "phrase_induce",
             "candidates": len(rep["candidates"])})
        lines.append("候选须经'说法验证 <id>'(独立验证例全过)再'说法激活 <id>'"
                     "才能参与解析。")
        return lines

    def _phrase_validate(self, args) -> list[str]:
        if not args:
            return ["用法: 说法验证 <规则id>"]
        from . import phraselearn as PL
        row = self.mem.get_phrase_rule(args[0])
        if not row:
            return ["查无此规则。先用'说法列表'查看。"]
        held = []
        for r in self.mem.list_phrase_examples(heldout=True):
            if PL._sig_key(PL.parse_for_filter(r["dst"],
                                               self._lexicon())) == \
                    PL._sig_key(row.get("signature", {})):
                held.append({"src": r["src"], "dst": r["dst"]})
        if not held:
            return [f"[phrase] {args[0]} 无同语义的验证例,无法验证。",
                    "用'说法 <新说法> => <同类标准说法> --留存'补充后再试。"]
        rep = PL.validate_rule(row, held, self._lexicon())
        if rep["passed"]:
            row["status"] = "validated"
            row["report"] = {**row.get("report", {}), **rep}
            self.mem.save_phrase_rule(row)
            self.mem.add_experience(
                "evaluation", f"phrase validate {args[0]}: passed",
                {"op": "phrase_validate", "report": rep})
            return [f"[phrase] {args[0]} 验证通过({rep['checked']}个验证例全中)。"
                    "状态validated；用'说法激活'显式激活后才参与解析。"]
        row["report"] = {**row.get("report", {}), **rep}
        self.mem.save_phrase_rule(row)
        bad = [d for d in rep.get("detail", []) if not d["ok"]]
        return [f"[phrase] {args[0]} 验证未通过: {rep.get('reason')}。",
                f"首个失败: {bad[0] if bad else rep}。规则停在candidate。"]

    def _phrase_activate(self, args) -> list[str]:
        if not args:
            return ["用法: 说法激活 <规则id>"]
        from . import phraselearn as PL
        out = PL.activate_rule(self.mem, args[0])
        if out.get("ok"):
            return [f"[phrase] {args[0]} 已激活,后续未知输入会尝试该改写"
                    "(直接能解析的输入不受影响)。"]
        return [f"[phrase] 激活失败: {out.get('error')}"]

    def _phrase_list(self, args) -> list[str]:
        rows = self.mem.list_phrase_rules()
        nex = len(self.mem.list_phrase_examples())
        out = [f"说法示例{nex}条；规则{len(rows)}条:"]
        for r in rows:
            out.append(f"  {r['id']} [{r['kind']}/{r['status']}] "
                       f"{r['src_tmpl']} -> {r['dst_tmpl']}")
        return out

    # ---- direct semantic teaching flow (Phase 6E): labeled examples ->
    # induce (candidate) -> held-out validate -> explicit activate.
    # Learned rules CONSTRUCT frames directly (no rewrite-and-reparse);
    # they only answer unknown inputs and never write knowledge.
    @staticmethod
    def _parse_sem_label(spec: str) -> dict | None:
        # kind=function_q;entity=杯子;key=function (no eval, fixed fields)
        out: dict = {}
        for chunk in spec.split(";"):
            chunk = chunk.strip()
            if not chunk or "=" not in chunk:
                return None
            k, v = chunk.split("=", 1)
            k, v = k.strip(), v.strip()
            if k not in ("kind", "entity", "key") or not v or k in out:
                return None
            out[k] = v
        return out if set(out) == {"kind", "entity", "key"} else None

    def _sem(self, args) -> list[str]:
        # 语义示例 <原句> => kind=<k>;entity=<e>;key=<k> [--留存]
        if "=>" not in " ".join(args):
            return ["用法: 语义示例 <原句> => kind=<用途/材质/存在性类>;entity=<对象>;"
                    "key=<属性键> [--留存]  (kinds: function_q/attr_q/has_q)"]
        heldout = args[-1] == "--留存"
        body = " ".join(args[:-1] if heldout else args)
        src, spec = (p.strip() for p in body.split("=>", 1))
        lab = self._parse_sem_label(spec)
        if lab is None:
            return ["[sem] 标注格式不对,须为 kind=<k>;entity=<e>;key=<k> "
                    "三字段(禁多余字段与表达式)。示例未收录。"]
        from . import semlearn as SL
        chk = SL.check_label(lab["kind"], lab["entity"], lab["key"])
        if not chk["ok"]:
            return [f"[sem] 示例未收录: {chk['reason']}。"]
        eid = self.mem.add_sem_example(
            src, lab["kind"], lab["entity"], lab["key"], heldout,
            evidence="chat-teach")
        self.mem.add_experience(
            "evaluation", f"chat sem example #{eid}",
            {"op": "sem_teach", "heldout": heldout})
        kind = "验证例" if heldout else "训练例"
        return [f"[sem] 已记{kind}#{eid}: {src!r} -> {lab['kind']}/"
                f"{lab['entity']}/{lab['key']}。",
                "示例只存不改解析；用'语义归纳'从训练例归纳候选结构。"]

    def _sem_induce(self, args) -> list[str]:
        from . import semlearn as SL
        trains = [r for r in self.mem.list_sem_examples(heldout=False)
                  if r.get("kind") != "capability"]
        if not trains:
            return ["训练例不足。先用'语义示例'教学(至少2个不同对象)。"]
        rep = SL.induce(
            [{"src": r["src"], "kind": r["kind"], "entity": r["entity"],
              "key": r["attrkey"]} for r in trains])
        lines = [f"[sem] 归纳完成: {len(rep['candidates'])} 候选, "
                 f"{len(rep['rejected'])} 组被拒(已说明原因,未建规则)。"]
        for c in rep["candidates"]:
            rid = SL.new_rule_id("chat")
            self.mem.save_sem_rule(
                {"id": rid, "kind": c["kind"], "pre": c["pre"],
                 "suf": c["suf"], "key": c["key"], "status": "candidate",
                 "evidence": [{"source": "chat", "support": c["support"]}],
                 "report": {"note": c["note"]}})
            lines.append(f"  {rid} [{c['kind']}/{c['key']}] {c['note']} "
                         f"(状态candidate,直接构帧,未验证未激活)")
        for r in rep["rejected"]:
            lines.append(f"  拒: {r.get('reason')}")
        self.mem.add_experience(
            "evaluation", "chat sem induce",
            {"op": "sem_induce", "candidates": len(rep["candidates"])})
        lines.append("候选须经'语义验证 <id>'(独立验证例全过)再'语义激活 <id>'"
                     "才能参与解析。")
        return lines

    def _sem_validate(self, args) -> list[str]:
        if not args:
            return ["用法: 语义验证 <规则id>"]
        from . import semlearn as SL
        row = self.mem.get_sem_rule(args[0])
        if not row:
            return ["查无此规则。先用'语义列表'查看。"]
        held = [{"src": r["src"], "entity": r["entity"]}
                for r in self.mem.list_sem_examples(heldout=True)
                if r["kind"] == row["kind"] and
                r["attrkey"] == row["key"]]
        if not held:
            return [f"[sem] {args[0]} 无同类验证例,无法验证。",
                    "用'语义示例 <句> => 同类标注 --留存'补充后再试。"]
        # 6F gate 1: label-vs-vocabulary consistency (detects labels
        # that contradict established word meanings, e.g. a 材质-form
        # labeled as a function question).
        vocab = SL.check_vocab_consistency(row)
        if not vocab["ok"]:
            row["report"] = {**row.get("report", {}), "vocab": vocab}
            self.mem.save_sem_rule(row)
            self.mem.add_experience(
                "evaluation", f"sem validate {args[0]}: vocab conflict",
                {"op": "sem_validate", "report": vocab})
            return [f"[sem] {args[0]} 验证未通过: {vocab['reason']}。",
                    "规则停在candidate；请检查标注后重新教学。"]
        # 6F gate 2: contrast silence over other-kind held-outs.
        contrast = [{"src": r["src"]}
                    for r in self.mem.list_sem_examples(heldout=True)
                    if (r["kind"], r["attrkey"]) !=
                    (row["kind"], row["key"])]
        rep = SL.validate_rule(row, held, contrast)
        row["report"] = {**row.get("report", {}), **rep}
        if rep["passed"]:
            row["status"] = "validated"
            self.mem.save_sem_rule(row)
            self.mem.add_experience(
                "evaluation", f"sem validate {args[0]}: passed",
                {"op": "sem_validate", "report": rep})
            return [f"[sem] {args[0]} 验证通过({rep['checked']}个验证例全中)。"
                    "状态validated；用'语义激活'显式激活后才参与解析。"]
        self.mem.save_sem_rule(row)
        bad = [d for d in rep.get("detail", []) if not d["ok"]]
        contra = rep.get("contrast", [])
        extra = f"对比冲突: {contra[0]}。" if contra else ""
        return [f"[sem] {args[0]} 验证未通过: {rep.get('reason')}。" + extra,
                f"首个失败: {bad[0] if bad else rep}。规则停在candidate。"]

    def _sem_activate(self, args) -> list[str]:
        if not args:
            return ["用法: 语义激活 <规则id>"]
        from . import semlearn as SL
        out = SL.activate_rule(self.mem, args[0])
        if out.get("ok"):
            return [f"[sem] {args[0]} 已激活,后续未知输入会尝试直接构帧"
                    "(直接能解析的输入不受影响)。"]
        return [f"[sem] 激活失败: {out.get('error')}"]

    def _sem_list(self, args) -> list[str]:
        rows = self.mem.list_sem_rules()
        nex = len(self.mem.list_sem_examples())
        out = [f"语义示例{nex}条；规则{len(rows)}条:"]
        for r in rows:
            out.append(f"  {r['id']} [{r['kind']}/{r['key']}/"
                       f"{r['status']}] {r['pre']!r} + ENTITY + "
                       f"{r['suf']!r}")
        return out

    # ---- bidirectional construction flow (Phase 6H-2B): examples ->
    # induce (candidate) -> held-out validate (parse + regenerate) ->
    # explicit activate. Parse and generate read the SAME stored row.
    @staticmethod
    def _parse_cx_label(spec: str) -> dict | None:
        # entity=<E>;value=<V> (no eval, fixed fields only)
        out: dict = {}
        for chunk in spec.split(";"):
            chunk = chunk.strip()
            if not chunk or "=" not in chunk:
                return None
            k, v = chunk.split("=", 1)
            k, v = k.strip(), v.strip()
            if k not in ("entity", "value") or not v or k in out:
                return None
            out[k] = v
        return out if set(out) == {"entity", "value"} else None

    def _construct(self, args) -> list[str]:
        # 构式 <原句> => entity=<E>;value=<V> [--留存]
        if "=>" not in " ".join(args):
            return ["用法: 构式 <原句> => entity=<实体>;value=<功能> [--留存]"]
        heldout = args[-1] == "--留存"
        body = " ".join(args[:-1] if heldout else args)
        src, spec = (p.strip() for p in body.split("=>", 1))
        lab = self._parse_cx_label(spec)
        if lab is None:
            return ["[construct] 标注格式不对,须为 entity=<实体>;value=<功能>。"
                    "示例未收录。"]
        try:
            eid = self.mem.add_construction_example(
                src, lab["entity"], lab["value"], heldout,
                evidence="chat-teach")
        except ValueError as e:
            return [f"[construct] 示例未收录: {e}。"]
        self.mem.add_experience(
            "evaluation", f"chat construction example #{eid}",
            {"op": "cx_teach", "heldout": heldout})
        kind = "验证例" if heldout else "训练例"
        return [f"[construct] 已记{kind}#{eid}: {src!r} -> "
                f"({lab['entity']}, {lab['value']})。",
                "示例只存不改解析；用'构式归纳'归纳候选构式。"]

    def _construct_induce(self, args) -> list[str]:
        from . import constructions as CX
        trains = [{"src": r["src"], "entity": r["entity"],
                   "value": r["value"]}
                  for r in self.mem.list_construction_examples(
                      heldout=False)]
        if not trains:
            return ["训练例不足。先用'构式'教学(至少2个不同主体)。"]
        rep = CX.induce(trains)
        lines = [f"[construct] 归纳完成: {len(rep['candidates'])} 候选, "
                 f"{len(rep['rejected'])} 组被拒(已说明原因,未建规则)。"]
        for c in rep["candidates"]:
            rid = CX.new_rule_id("chat")
            self.mem.save_construction(
                {"id": rid, "kind": c["kind"], "pre": c["pre"],
                 "mid": c["mid"], "suf": c["suf"],
                 "fixed_value": c.get("fixed_value", ""),
                 "status": "candidate",
                 "evidence": [{"source": "chat", "support": c["support"]}],
                 "report": {"note": c["note"]}})
            lines.append(f"  {rid} [{c['kind']}] {c['note']} "
                         f"(状态candidate,未验证未激活)")
        for r in rep["rejected"]:
            lines.append(f"  拒: {r.get('reason')}")
        self.mem.add_experience(
            "evaluation", "chat construction induce",
            {"op": "cx_induce", "candidates": len(rep["candidates"])})
        lines.append("候选须经'构式验证 <id>'(独立验证例双向全过)再"
                     "'构式激活 <id>'才能使用。")
        return lines

    def _construct_validate(self, args) -> list[str]:
        if not args:
            return ["用法: 构式验证 <规则id>"]
        from . import constructions as CX
        row = self.mem.get_construction(args[0])
        if not row:
            return ["查无此规则。先用'构式列表'查看。"]
        held = [{"src": r["src"], "entity": r["entity"],
                 "value": r["value"]}
                for r in self.mem.list_construction_examples(heldout=True)]
        if not held:
            return [f"[construct] {args[0]} 无验证例,无法验证。",
                    "用'构式 <句> => 标注 --留存'补充后再试。"]
        rep = CX.validate_rule(row, held)
        row["report"] = {**row.get("report", {}), **rep}
        if rep["passed"]:
            row["status"] = "validated"
            self.mem.save_construction(row)
            self.mem.add_experience(
                "evaluation", f"construction validate {args[0]}: passed",
                {"op": "cx_validate", "report": rep})
            return [f"[construct] {args[0]} 验证通过({rep['checked']}个验证例"
                    "解析+重生成全中)。状态validated；用'构式激活'显式激活。"]
        self.mem.save_construction(row)
        bad = [d for d in rep.get("detail", []) if not d["ok"]]
        return [f"[construct] {args[0]} 验证未通过: {rep.get('reason')}。",
                f"首个失败: {bad[0] if bad else rep}。规则停在candidate。"]

    def _construct_activate(self, args) -> list[str]:
        if not args:
            return ["用法: 构式激活 <规则id>"]
        from . import constructions as CX
        out = CX.activate_rule(self.mem, args[0])
        if out.get("ok"):
            return [f"[construct] {args[0]} 已激活,解析与生成共用该定义。"]
        return [f"[construct] 激活失败: {out.get('error')}"]

    def _construct_list(self, args) -> list[str]:
        rows = self.mem.list_constructions()
        nex = len(self.mem.list_construction_examples())
        out = [f"构式示例{nex}条；规则{len(rows)}条:"]
        for r in rows:
            out.append(f"  {r['id']} [{r['kind']}/{r['status']}] "
                       f"{r['pre']!r} + ENTITY + {r['mid']!r} + VALUE + "
                       f"{r['suf']!r}")
        return out

    def _fallback(self, line: str) -> list[str]:
        # try compositional parse before giving up; a parsed frame still
        # needs explicit 确认 (use 理解 to keep one entry point)
        try:
            fr = parse_sentence(line, source="user",
                               lexicon=self._lexicon(),
                               phrase_rules=self._phrase_rules())
        except Exception:
            fr = None
        if fr is not None and fr.intent != "unknown":
            return ["这句话不在命令表里,但我初步解析出结构:",
                    f"意图={fr.intent} 模态={fr.modality} 实体={fr.entities}",
                    "请用理解命令走确认流程(确认前不写入)。"]
        parsed = parse_chinese_simple(line)
        if parsed.get("parsed"):
            k = parsed["kind"]
            if k == "is_a":
                return ["我识别出一种归属说法,但不能仅凭一句话改结构。",
                        f"解析: {parsed}",
                        f"请用'观察 {parsed['subj']} ...'补充属性,或用'假设'存为待验证想法。"]
            return ["我识别出一种属性/能力说法,但需要结构化确认。",
                    f"解析: {parsed}",
                    "请用'观察 <名称> 属性=值'的格式重述,或用'假设'暂存。"]
        return ["暂时无法理解这句话(只支持帮助中列出的命令与常见陈述句式)。",
                "请换一种表达,或用'观察 <名称> 属性=值'结构化输入。",
                "本次未写入任何知识,未修改模型。"]
