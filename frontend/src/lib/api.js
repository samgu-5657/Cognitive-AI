// Thin client for the Python web API (src/cognitive_ai/web_ui.py).
// Every endpoint answers `{ok, ...}`; failures carry `error`, and the
// confirm/reject handlers report business failures as `{ok:false, reason}`.

export class ApiError extends Error {
  constructor(message, { status = 0, payload = null, offline = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
    this.offline = offline;
  }
}

export function createApi(fetchImpl = (...args) => globalThis.fetch(...args), base = "") {
  async function request(method, path, body) {
    let res;
    try {
      res = await fetchImpl(base + path, {
        method,
        headers: method === "POST" ? { "Content-Type": "application/json" } : undefined,
        body: method === "POST" ? JSON.stringify(body ?? {}) : undefined,
      });
    } catch (err) {
      throw new ApiError("无法连接服务器，请确认后端已启动", { offline: true });
    }
    let data;
    try {
      data = await res.json();
    } catch {
      throw new ApiError(`服务器返回了非 JSON 响应（HTTP ${res.status}）`, { status: res.status });
    }
    if (!data || data.ok !== true) {
      const msg = data?.error || data?.reason || `请求失败（HTTP ${res.status}）`;
      throw new ApiError(msg, { status: res.status, payload: data, offline: !!data?.offline });
    }
    return data;
  }

  const get = (path) => request("GET", path);
  const post = (path, body) => request("POST", path, body);

  return {
    health: () => get("/api/health"),
    status: () => get("/api/status"),
    pending: () => get("/api/pending"),
    ask: (text) => post("/api/ask", { text }),
    analyze: (text) => post("/api/analyze", { text }),
    context: (text, priors) => post("/api/context", { text, priors }),
    interact: (text, priors) => post("/api/interact", { text, priors }),
    answer: (request, optionId, priors, text) =>
      post("/api/answer", {
        request,
        answer: { request_id: request.request_id, option_id: optionId },
        priors,
        text,
      }),
    propose: (text) => post("/api/propose", { text }),
    confirm: (id) => post("/api/confirm", { id }),
    reject: (id) => post("/api/reject", { id }),
  };
}

export const api = createApi();
