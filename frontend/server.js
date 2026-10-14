// Bun dev/prod server: serves the React app and proxies /api/* to the
// Python backend (`python -m cognitive_ai.cli ui`, default 127.0.0.1:8765).
import index from "./src/index.html";

const API = (process.env.COGNITIVE_AI_API || "http://127.0.0.1:8765").replace(/\/$/, "");
const HOST = process.env.HOST || "127.0.0.1";
const PORT = Number(process.env.PORT || 5173);
const dev = process.env.NODE_ENV !== "production";

async function proxy(req) {
  const url = new URL(req.url);
  const target = API + url.pathname + url.search;
  try {
    const res = await fetch(target, {
      method: req.method,
      headers: { "Content-Type": "application/json" },
      body: req.method === "GET" || req.method === "HEAD" ? undefined : await req.text(),
    });
    return new Response(res.body, {
      status: res.status,
      headers: { "Content-Type": res.headers.get("Content-Type") || "application/json; charset=utf-8" },
    });
  } catch (err) {
    return Response.json(
      { ok: false, error: `无法连接后端 ${API}（${err?.code || err?.message || "unknown"}）`, offline: true },
      { status: 502 },
    );
  }
}

const server = Bun.serve({
  hostname: HOST,
  port: PORT,
  development: dev ? { hmr: true, console: true } : false,
  routes: {
    "/api/*": proxy,
    "/*": index,
  },
});

console.log(`Cognitive AI frontend: ${server.url}  (API → ${API})`);
