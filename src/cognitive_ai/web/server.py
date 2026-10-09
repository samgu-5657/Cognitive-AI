"""Local workbench HTTP server (Phase 5A). localhost only, no CDN, no
external calls. All routes delegate to service.Workbench (real backend).
Run: py -m cognitive_ai.cli ui  (or python -m cognitive_ai.web.server)"""
from __future__ import annotations

import os

try:
    # module level so FastAPI can resolve annotations (the file uses
    # `from __future__ import annotations`; function-local imports would
    # degrade Request params to query params with 422s)
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    _HAVE_FASTAPI = True
except ImportError:
    _HAVE_FASTAPI = False

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "static")


def _require_web():
    if not _HAVE_FASTAPI:
        raise RuntimeError(
            "web UI needs the optional 'ui' extra: "
            "py -m pip install -e \".[ui]\". The CLI works without it.")


def create_app(data_dir: str, models_root: str | None = None):
    _require_web()
    from .service import Workbench

    wb = Workbench(data_dir, models_root)
    app = FastAPI(title="Cognitive AI Workbench", version="5A")
    app.state.wb = wb

    def ok(data):
        return {"ok": True, "data": data}

    def fail(msg: str, code: int = 400):
        return JSONResponse({"ok": False, "error": str(msg)},
                            status_code=code)

    @app.get("/api/status")
    def api_status():
        try:
            return ok(wb.status())
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/chat")
    async def api_chat(req: Request):
        try:
            b = await req.json()
            return ok(wb.chat(b.get("text", ""), b.get("session_id") or "web"))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/solve")
    async def api_solve(req: Request):
        try:
            b = await req.json()
            return ok(wb.solve(b.get("text", ""), b.get("session_id") or "web",
                               b.get("style") or "short"))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/understand")
    async def api_understand(req: Request):
        try:
            b = await req.json()
            return ok(wb.understand(b.get("text", ""),
                                    b.get("session_id") or "web"))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/confirm")
    async def api_confirm(req: Request):
        try:
            b = await req.json()
            return ok(wb.confirm(b.get("session_id") or "web"))
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/teach/preview")
    async def api_teach_preview(req: Request):
        try:
            b = await req.json()
            return ok(wb.teach_preview(b.get("name", ""),
                                       b.get("attributes", {}) or {},
                                       b.get("parents", []) or []))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/teach/confirm")
    async def api_teach_confirm(req: Request):
        try:
            b = await req.json()
            return ok(wb.teach_confirm(
                b.get("name", ""), b.get("attributes", {}) or {},
                b.get("parents", []) or [], b.get("source") or "web"))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/correct")
    async def api_correct(req: Request):
        try:
            b = await req.json()
            return ok(wb.correct(b.get("name", ""), b.get("key", ""),
                                 b.get("value", ""), b.get("reason", "")))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/api/rules")
    def api_rules(status: str | None = None, domain: str | None = None):
        try:
            return ok(wb.rule_list(status=status, domain=domain))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/api/rules/{rid}")
    def api_rule_show(rid: str):
        try:
            return ok(wb.rule_show(rid))
        except ValueError as e:
            return fail(e, 404)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/rules/induce")
    async def api_rule_induce(req: Request):
        try:
            b = await req.json()
            return ok(wb.rule_induce(b.get("name", ""), b.get("domain", ""),
                                     b.get("examples", []) or [],
                                     b.get("negatives", []) or []))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/rules/select")
    async def api_rule_select(req: Request):
        try:
            b = await req.json()
            return ok(wb.rule_select(b.get("rule_id", ""), b.get("index", 0),
                                     b.get("reason", "")))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/rules/validate")
    async def api_rule_validate(req: Request):
        try:
            b = await req.json()
            return ok(wb.rule_validate(
                b.get("rule_id", ""), b.get("train", []) or [],
                b.get("heldout", []) or [], b.get("negatives", []) or [],
                bool(b.get("activate"))))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/rules/rollback")
    async def api_rule_rollback(req: Request):
        try:
            b = await req.json()
            return ok(wb.rule_rollback(b.get("rule_id", "")))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/api/rules/{rid}/calls")
    def api_rule_calls(rid: str):
        try:
            return ok(wb.rule_calls(rid))
        except ValueError as e:
            return fail(e, 404)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/api/concepts")
    def api_concepts(q: str = "", kind: str = ""):
        try:
            return ok(wb.concepts(q[:100], kind[:50]))
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/api/concepts/{cid}")
    def api_concept(cid: str):
        try:
            return ok(wb.concept_detail(cid))
        except ValueError as e:
            return fail(e, 404)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/api/queue")
    def api_queue(status: str | None = None):
        try:
            return ok(wb.queue(status=status))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/queue/review")
    async def api_queue_review(req: Request):
        try:
            b = await req.json()
            return ok(wb.queue_review(b.get("sample_id", ""),
                                      bool(b.get("accept")),
                                      b.get("l1"), b.get("l2")))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/explore")
    async def api_explore(req: Request):
        try:
            b = await req.json()
            return ok(wb.explore(b.get("steps", 3)))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/api/gaps")
    def api_gaps():
        try:
            return ok(wb.gaps())
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/api/history")
    def api_history(limit: int = 50):
        try:
            return ok(wb.history(max(1, min(limit, 200))))
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/api/tasks")
    def api_tasks(limit: int = 20):
        try:
            return ok(wb.tasks(limit))
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/api/tasks/{tid}")
    def api_task(tid: str):
        try:
            return ok(wb.task_detail(tid))
        except ValueError as e:
            return fail(e, 404)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/api/analogies")
    def api_analogies(limit: int = 30):
        try:
            return ok(wb.analogies(limit))
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/analogies/correct")
    async def api_analogy_correct(req: Request):
        try:
            b = await req.json()
            return ok(wb.analogy_correct(b.get("task_id", ""),
                                         b.get("candidate_task_id", ""),
                                         b.get("effect", "")))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/api/model")
    def api_model():
        try:
            return ok(wb.model_status())
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.post("/api/predict")
    async def api_predict(req: Request):
        try:
            b = await req.json()
            return ok(wb.predict(b.get("text", "")))
        except ValueError as e:
            return fail(e)
        except Exception as e:  # noqa: BLE001
            return fail(e, 500)

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main() -> None:
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost"):
        print("拒绝: UI 只允许绑定本地回环地址。", file=sys.stderr)
        raise SystemExit(2)
    if not (1 <= args.port <= 65535):
        print("拒绝: 端口超出范围。", file=sys.stderr)
        raise SystemExit(2)
    _require_web()
    import uvicorn
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "..", ".."))
    data = args.data or os.path.join(here, "data")
    models = args.models or os.path.join(here, "models")
    os.makedirs(data, exist_ok=True)
    print(f"Cognitive AI 工作台启动中…\n  地址: http://{args.host}:{args.port}/\n"
          f"  数据目录: {data}\n  模型目录: {models}")
    os.environ["COG_DATA"] = data
    os.environ["COG_MODELS"] = models
    app = create_app(data, models)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except OSError as e:
        print(f"启动失败(端口可能被占用): {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
