"""Local read-mostly web UI (M5-C-UI-01). Standard library only.

Every route calls the real cognitive modules (semground, semmeaning,
semcontext, Learner); handlers never re-implement parsing, grounding
or gating. Sessions hold no server state: the browser sends explicit
priors with each context request. Writes happen only through explicit
propose/confirm/reject calls. Binds 127.0.0.1 by default.
"""
from __future__ import annotations

import dataclasses
import json
import os
import threading
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from urllib.parse import urlparse

STATIC_DIR = os.path.join(os.path.dirname(__file__), "web_ui_static")


def json_safe(obj):
    """Strict structural serializer. Unknown types raise, never str()."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, MappingProxyType):
        return {k: json_safe(v) for k, v in dict(obj).items()}
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((json_safe(v) for v in obj),
                      key=lambda v: json.dumps(v, ensure_ascii=False,
                                               sort_keys=True))
    if dataclasses.is_dataclass(obj):
        return {"__type__": type(obj).__name__,
                **{f.name: json_safe(getattr(obj, f.name))
                   for f in dataclasses.fields(obj)}}
    raise TypeError(f"not serializable: {type(obj).__name__}")


def _need_text(data: dict) -> str:
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请输入非空文本")
    return text


def handle_health(_mem) -> dict:
    return {"ok": True, "service": "cognitive-ai-ui"}


def handle_status(mem) -> dict:
    from cognitive_ai.ontology import Ontology
    onto = Ontology(mem)
    return {"concepts": len(onto.list()),
            "rules": len(mem.list_rules()),
            "pending": len(mem.list_pending_candidates(status="proposed")),
            "version": mem.current_version()}


def handle_ask(mem, text: str) -> dict:
    from cognitive_ai.semground import answer_question
    from cognitive_ai.semrealize import build_answer_semantics, realize
    doc = answer_question(text, mem).to_dict()
    sem = build_answer_semantics(doc)
    out = realize(sem)
    return {"verdict": doc, "answer": out["text"],
            "constructions": out.get("constructions", []),
            "roundtrippable": out.get("roundtrippable", False)}


def handle_analyze(mem, text: str) -> dict:
    from cognitive_ai.semmeaning import understand_compositional_statement
    rep = understand_compositional_statement(text, mem)
    return {"report": rep}


def handle_context(mem, priors, text: str) -> dict:
    from cognitive_ai.semcontext import (empty_context,
                                         extend_semantic_context,
                                         understand_with_context)
    from cognitive_ai.semmeaning import understand_compositional_statement
    if not isinstance(priors, list) or \
            any(not isinstance(p, str) for p in priors):
        raise ValueError("priors 须为字符串数组")
    context = empty_context()
    for prior in priors:
        if not prior.strip():
            continue
        analyzed = understand_compositional_statement(prior, mem)
        context = extend_semantic_context(context, analyzed, mem)
    rep = understand_with_context(text, mem, context)
    return {"context": context, "report": rep}


def handle_interact(mem, priors, text: str) -> dict:
    from cognitive_ai import seminteraction as inter
    from cognitive_ai.semcontext import (empty_context,
                                         extend_semantic_context)
    from cognitive_ai.semmeaning import understand_compositional_statement
    if not isinstance(priors, list) or \
            any(not isinstance(p, str) for p in priors):
        raise ValueError("priors 须为字符串数组")
    context = empty_context()
    for prior in priors:
        if not prior.strip():
            continue
        analyzed = understand_compositional_statement(prior, mem)
        context = extend_semantic_context(context, analyzed, mem)
    return {"result": inter.begin_semantic_interaction(text, mem,
                                                       context)}


def handle_answer_clarification(mem, request_data: dict,
                                answer_data: dict, priors,
                                text: str) -> dict:
    from cognitive_ai import seminteraction as inter
    from cognitive_ai.semcontext import (empty_context,
                                         extend_semantic_context)
    from cognitive_ai.semmeaning import understand_compositional_statement
    from cognitive_ai.models import SemanticClarificationAnswer
    if not isinstance(request_data, dict) or \
            not isinstance(answer_data, dict):
        raise ValueError("请求与回答须为对象")
    if not isinstance(priors, list) or \
            any(not isinstance(p, str) for p in priors):
        raise ValueError("priors 须为字符串数组")
    request = inter.clarification_request_from_json(request_data)
    option_id = answer_data.get("option_id", "")
    if not isinstance(option_id, str) or not option_id:
        raise ValueError("选项 ID 须为非空字符串")
    if answer_data.get("request_id", "") != request.request_id:
        raise ValueError("回答与请求不匹配")
    context = empty_context()
    for prior in priors:
        if not prior.strip():
            continue
        analyzed = understand_compositional_statement(prior, mem)
        context = extend_semantic_context(context, analyzed, mem)
    answer = SemanticClarificationAnswer(
        request_id=request.request_id, option_id=option_id,
        context_fingerprint=inter.context_fingerprint(context),
        origin="user")
    return {"result": inter.answer_semantic_clarification(
        request, context, answer, mem)}
    from cognitive_ai.learner import Learner
    cand = Learner(mem, corpus_dir).propose_from_utterance(text)
    return {"candidate": cand}


def handle_propose(mem, text: str, corpus_dir: str = "") -> dict:
    from cognitive_ai.learner import Learner
    cand = Learner(mem, corpus_dir).propose_from_utterance(text)
    return {"candidate": cand}


def handle_pending(mem, status=None) -> dict:
    from cognitive_ai.learner import Learner
    return {"candidates": Learner(mem, "").list_pending_candidates(status)}


def handle_confirm(mem, pid: int) -> dict:
    from cognitive_ai.learner import Learner
    row = mem.get_pending_candidate(int(pid))
    if row is None:
        raise LookupError(f"候选 #{pid} 不存在")
    return Learner(mem, "").confirm_candidate(int(pid))


def handle_reject(mem, pid: int) -> dict:
    from cognitive_ai.learner import Learner
    row = mem.get_pending_candidate(int(pid))
    if row is None:
        raise LookupError(f"候选 #{pid} 不存在")
    return Learner(mem, "").reject_candidate(int(pid))


_ROUTES = {
    ("GET", "/api/health"): ("health", False),
    ("GET", "/api/status"): ("status", False),
    ("GET", "/api/pending"): ("pending", False),
    ("POST", "/api/ask"): ("ask", True),
    ("POST", "/api/analyze"): ("analyze", True),
    ("POST", "/api/context"): ("context", True),
    ("POST", "/api/interact"): ("interact", True),
    ("POST", "/api/answer"): ("answer", True),
    ("POST", "/api/propose"): ("propose", True),
    ("POST", "/api/confirm"): ("confirm", True),
    ("POST", "/api/reject"): ("reject", True),
}


def dispatch(mem, corpus_dir: str, method: str, path: str,
             body: dict):
    """Route one request. Returns (status_code, payload_dict)."""
    try:
        route = _ROUTES.get((method, path))
        if route is None:
            return 404, {"ok": False, "error": "未知接口"}
        name, needs_body = route
        if needs_body and not isinstance(body, dict):
            return 400, {"ok": False, "error": "请求体须为 JSON 对象"}
        if name == "health":
            return 200, {"ok": True, **handle_health(mem)}
        if name == "status":
            return 200, {"ok": True, **handle_status(mem)}
        if name == "pending":
            return 200, {"ok": True,
                         **handle_pending(mem, (body or {}).get("status"))}
        if name in ("ask", "analyze", "propose"):
            text = _need_text(body)
            if name == "ask":
                return 200, {"ok": True, **handle_ask(mem, text)}
            if name == "analyze":
                return 200, {"ok": True, **handle_analyze(mem, text)}
            return 200, {"ok": True,
                         **handle_propose(mem, text, corpus_dir)}
        if name == "context":
            text = _need_text(body)
            priors = body.get("priors", [])
            return 200, {"ok": True,
                         **handle_context(mem, priors, text)}
        if name == "interact":
            text = _need_text(body)
            priors = body.get("priors", [])
            return 200, {"ok": True,
                         **handle_interact(mem, priors, text)}
        if name == "answer":
            priors = body.get("priors", [])
            text = body.get("text", "")
            if not isinstance(text, str) or not text.strip():
                return 400, {"ok": False, "error": "请输入非空文本"}
            return 200, {"ok": True,
                         **handle_answer_clarification(
                             mem, body.get("request", {}),
                             body.get("answer", {}), priors, text)}
        pid = (body or {}).get("id")
        if not isinstance(pid, int):
            return 400, {"ok": False, "error": "候选 id 须为整数"}
        if name == "confirm":
            return 200, {"ok": True, **handle_confirm(mem, pid)}
        return 200, {"ok": True, **handle_reject(mem, pid)}
    except (ValueError, LookupError) as e:
        code = 404 if isinstance(e, LookupError) else 400
        return code, {"ok": False, "error": str(e)}
    except Exception as e:
        return 500, {"ok": False, "error": f"内部错误：{type(e).__name__}"}


class _Handler(BaseHTTPRequestHandler):
    mem = None
    corpus_dir = ""
    server_version = "CognitiveAI/1.0"

    def log_message(self, *args):
        pass

    def _send(self, code: int, payload: dict) -> None:
        try:
            body = json.dumps(json_safe(payload), ensure_ascii=False)
        except TypeError as e:
            code, body = 500, json.dumps(
                {"ok": False, "error": f"序列化失败：{e}"},
                ensure_ascii=False)
        raw = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _mem(self):
        from cognitive_ai import cli as _cli
        return _cli.get_memory()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_file("index.html", "text/html; charset=utf-8")
            return
        mem = self._mem()
        try:
            if parsed.path == "/api/pending":
                code, payload = dispatch(mem, self.corpus_dir, "GET",
                                         "/api/pending", {})
            elif parsed.path == "/api/health":
                code, payload = dispatch(mem, self.corpus_dir, "GET",
                                         "/api/health", {})
            elif parsed.path == "/api/status":
                code, payload = dispatch(mem, self.corpus_dir, "GET",
                                         "/api/status", {})
            else:
                code, payload = 404, {"ok": False, "error": "未知接口"}
        finally:
            mem.close()
        self._send(code, payload)

    def _send_file(self, name: str, mime: str) -> None:
        path = os.path.join(STATIC_DIR, name)
        if os.path.basename(path) != name or \
                not os.path.isfile(path):
            self._send(404, {"ok": False, "error": "页面不存在"})
            return
        with open(path, "rb") as fh:
            raw = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        try:
            body = json.loads(
                self.rfile.read(length).decode("utf-8") or "{}")
        except ValueError:
            self._send(400, {"ok": False, "error": "请求体不是合法 JSON"})
            return
        mem = self._mem()
        try:
            code, payload = dispatch(mem, self.corpus_dir,
                                     "POST", parsed.path, body)
        finally:
            mem.close()
        self._send(code, payload)


def run(host: str = "127.0.0.1", port: int = 8765,
        corpus_dir: str = ""):
    """Start the local UI server (blocking). Localhost by default."""
    server = ThreadingHTTPServer((host, port), _Handler)
    _Handler.corpus_dir = corpus_dir
    print(f"Cognitive AI UI: http://{host}:{server.server_address[1]}",
          flush=True)
    server.serve_forever()


def run_in_thread(host: str = "127.0.0.1", port: int = 0,
                  corpus_dir: str = ""):
    """Start the server in a daemon thread; returns (server, thread)."""
    server = ThreadingHTTPServer((host, port), _Handler)
    _Handler.corpus_dir = corpus_dir
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
