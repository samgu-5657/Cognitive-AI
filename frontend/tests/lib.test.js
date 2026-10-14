import { describe, expect, test } from "bun:test";
import { createApi, ApiError } from "../src/lib/api.js";
import {
  reducer,
  initialState,
  newConversation,
  loadState,
  saveState,
  titleFrom,
  STORAGE_KEY,
  MAX_CONVERSATIONS,
} from "../src/lib/conversations.js";
import {
  summarizeAsk,
  buildTree,
  summarizeContext,
  priorFromInteraction,
  summarizeCandidate,
  countByStatus,
  describeDecision,
  describeFailure,
  translateReason,
} from "../src/lib/summarize.js";
import { statusInfo } from "../src/lib/labels.js";

function memoryStorage(limit = Infinity) {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => {
      if (String(v).length > limit) throw new Error("QuotaExceededError");
      m.set(k, String(v));
    },
    removeItem: (k) => m.delete(k),
  };
}

const fakeFetch = (status, body, seen = []) => async (url, init) => {
  seen.push({ url, init });
  return { status, json: async () => (typeof body === "function" ? body() : body) };
};

describe("api client", () => {
  test("posts JSON bodies to the expected endpoints", async () => {
    const seen = [];
    const api = createApi(fakeFetch(200, { ok: true, answer: "是的" }, seen));
    const out = await api.ask("杯子是不是容器？");
    expect(out.answer).toBe("是的");
    expect(seen[0].url).toBe("/api/ask");
    expect(seen[0].init.method).toBe("POST");
    expect(JSON.parse(seen[0].init.body)).toEqual({ text: "杯子是不是容器？" });

    await api.interact("它可以盛水", ["杯子是容器"]);
    expect(JSON.parse(seen[1].init.body)).toEqual({ text: "它可以盛水", priors: ["杯子是容器"] });

    await api.pending();
    expect(seen[2].init.method).toBe("GET");
    expect(seen[2].init.body).toBeUndefined();
  });

  test("answer() echoes the request id the backend requires", async () => {
    const seen = [];
    const api = createApi(fakeFetch(200, { ok: true, result: {} }, seen));
    const req = { request_id: "clr-1", options: [] };
    await api.answer(req, "opt-cup", ["a"], "它");
    expect(JSON.parse(seen[0].init.body)).toEqual({
      request: req,
      answer: { request_id: "clr-1", option_id: "opt-cup" },
      priors: ["a"],
      text: "它",
    });
  });

  test("surfaces backend `error` and confirm/reject `reason` failures", async () => {
    const a = createApi(fakeFetch(400, { ok: false, error: "请输入非空文本" }));
    await expect(a.ask(" ")).rejects.toThrow("请输入非空文本");
    // confirm on an already-decided candidate: HTTP 200 but ok:false + reason
    const b = createApi(fakeFetch(200, { ok: false, reason: "candidate is accepted, not proposed" }));
    const err = await b.confirm(1).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toBe("candidate is accepted, not proposed");
  });

  test("marks network failures as offline", async () => {
    const api = createApi(async () => {
      throw new TypeError("fetch failed");
    });
    const err = await api.status().catch((e) => e);
    expect(err.offline).toBe(true);
  });

  test("rejects non-JSON responses", async () => {
    const api = createApi(fakeFetch(502, () => {
      throw new SyntaxError("bad json");
    }));
    await expect(api.status()).rejects.toThrow("HTTP 502");
  });
});

describe("conversation reducer", () => {
  const user = (text) => ({ id: "u-" + text, role: "user", mode: "ask", text });
  const asst = (id) => ({ id, role: "assistant", mode: "ask", status: "loading" });

  test("send creates the conversation on first message and titles it", () => {
    let s = reducer(initialState(), { type: "send", convId: "c1", user: user("杯子是不是容器？"), assistant: asst("a1") });
    expect(s.activeId).toBe("c1");
    expect(s.conversations).toHaveLength(1);
    expect(s.conversations[0].title).toBe("杯子是不是容器？");
    s = reducer(s, { type: "send", convId: "c1", user: user("第二句"), assistant: asst("a2") });
    expect(s.conversations[0].messages).toHaveLength(4);
    expect(s.conversations[0].title).toBe("杯子是不是容器？");
  });

  test("patchData merges instead of overwriting earlier patches", () => {
    let s = reducer(initialState(), { type: "send", convId: "c1", user: user("x"), assistant: asst("a1") });
    s = reducer(s, { type: "patchMessage", convId: "c1", msgId: "a1", patch: { status: "done", data: { candidate: { id: 3 } } } });
    s = reducer(s, { type: "patchData", convId: "c1", msgId: "a1", patch: { decision: { ok: true } } });
    s = reducer(s, { type: "patchData", convId: "c1", msgId: "a1", patch: { deciding: false } });
    const m = s.conversations[0].messages[1];
    expect(m.data).toEqual({ candidate: { id: 3 }, decision: { ok: true }, deciding: false });
  });

  test("new reuses an untouched empty conversation", () => {
    let s = reducer(initialState(), { type: "new" });
    const first = s.activeId;
    s = reducer(s, { type: "new" });
    expect(s.conversations).toHaveLength(1);
    expect(s.activeId).toBe(first);
  });

  test("delete moves focus and restore brings it back", () => {
    let s = reducer(initialState(), { type: "send", convId: "c1", user: user("a"), assistant: asst("a1") });
    s = reducer(s, { type: "send", convId: "c2", user: user("b"), assistant: asst("a2") });
    const removed = s.conversations.find((c) => c.id === "c2");
    s = reducer(s, { type: "delete", id: "c2" });
    expect(s.activeId).toBe("c1");
    s = reducer(s, { type: "restore", conversation: removed });
    expect(s.activeId).toBe("c2");
    expect(s.conversations).toHaveLength(2);
  });

  test("setPriors drops blank statements", () => {
    let s = reducer(initialState(), { type: "restore", conversation: { ...newConversation(), id: "c" } });
    s = reducer(s, { type: "setPriors", convId: "c", priors: ["杯子是容器", "  ", ""] });
    expect(s.conversations[0].priors).toEqual(["杯子是容器"]);
  });

  test("caps the number of stored conversations", () => {
    let s = initialState();
    for (let i = 0; i < MAX_CONVERSATIONS + 5; i++)
      s = reducer(s, { type: "send", convId: "c" + i, user: user(String(i)), assistant: asst("a" + i) });
    expect(s.conversations).toHaveLength(MAX_CONVERSATIONS);
  });

  test("titleFrom trims and truncates", () => {
    expect(titleFrom("   ")).toBe("新对话");
    expect(titleFrom("a".repeat(30))).toBe("a".repeat(24) + "…");
  });
});

describe("persistence", () => {
  test("round-trips and settles requests that never finished", () => {
    const st = memoryStorage();
    let s = reducer(initialState(), {
      type: "send",
      convId: "c1",
      user: { id: "u", role: "user", mode: "ask", text: "x" },
      assistant: { id: "a", role: "assistant", mode: "ask", status: "loading" },
    });
    expect(saveState(st, s)).toBe(true);
    const back = loadState(st);
    expect(back.activeId).toBe("c1");
    expect(back.conversations[0].messages[1].status).toBe("error");
  });

  test("tolerates corrupt or missing storage", () => {
    const st = memoryStorage();
    st.setItem(STORAGE_KEY, "{not json");
    expect(loadState(st)).toEqual(initialState());
    expect(loadState(null)).toEqual(initialState());
    expect(saveState(null, initialState())).toBe(false);
  });

  test("drops old conversations when the quota is exceeded", () => {
    let s = initialState();
    for (let i = 0; i < 8; i++)
      s = reducer(s, {
        type: "send",
        convId: "c" + i,
        now: i,
        user: { id: "u" + i, role: "user", mode: "ask", text: "x".repeat(200) },
        assistant: { id: "a" + i, role: "assistant", mode: "ask", status: "done" },
      });
    const st = memoryStorage(1500);
    expect(saveState(st, s)).toBe(true);
    const back = loadState(st);
    expect(back.conversations.length).toBeLessThan(8);
    expect(back.conversations[0].id).toBe("c7"); // newest kept
  });
});

describe("summaries", () => {
  // Shapes copied from real /api responses (see web_ui.py).
  const ask = {
    answer: "是的，杯子是容器。（已验证，依据：知识库记录）",
    verdict: {
      verdict: "supported",
      subject_match: { text: "杯子", concept_id: "cup" },
      object_match: { text: "容器", concept_id: "container" },
      queries: [{ op: "is_a", concept: "cup", ancestor: "容器", negated: false }],
      evidence: [{ kind: "is_a", ref: "cup->container", status: "verified", detail: "" }],
      detail: "decided hierarchy",
      question: { qkind: "is_a_verify", unknown_spans: ["容器"] },
    },
    constructions: ["verify:is_a"],
    roundtrippable: true,
  };

  test("summarizeAsk keeps the backend's verdict and evidence", () => {
    const s = summarizeAsk(ask);
    expect(s.verdict).toBe("supported");
    expect(s.subject.concept_id).toBe("cup");
    expect(s.evidence).toHaveLength(1);
    expect(s.qkind).toBe("is_a_verify");
    // object_match {} must not render as a tag
    expect(summarizeAsk({ verdict: { object_match: {} } }).object).toBeNull();
  });

  test("buildTree pairs evaluation nodes with parse roles", () => {
    const tree = buildTree(
      {
        kind: "conditional",
        raw_text: "如果A，那么B",
        status: "conditional_unasserted",
        children: [
          { kind: "atom", raw_text: "A", status: "supported", grounding: { verdict: "supported", evidence: [] }, children: [] },
          { kind: "atom", raw_text: "B", status: "unknown", grounding: {}, children: [] },
        ],
      },
      {
        kind: "conditional",
        roles: ["antecedent", "consequent"],
        connective: "",
        children: [
          { kind: "atom", atom: { subject: "杯子", predicate: "is_a", object: "容器" }, children: [] },
          { kind: "atom", atom: {}, children: [] },
        ],
      },
    );
    expect(tree.children.map((c) => c.role)).toEqual(["antecedent", "consequent"]);
    expect(tree.children[0].atom.subject).toBe("杯子");
    expect(tree.children[1].atom).toBeNull();
  });

  test("context resolutions and prior extraction", () => {
    const ctx = summarizeContext({
      raw_input: "它可以盛水",
      normalized_text: "杯子可以盛水",
      resolutions: [{ mention: "它", status: "resolved", resolved_ref: { name: "杯子", ref_id: "cup" }, candidates: [] }],
    });
    expect(ctx.resolutions[0]).toMatchObject({ mention: "它", target: "杯子", status: "resolved" });

    expect(priorFromInteraction({ state: "understood", understanding: { raw_input: "杯子可以盛水" } }, "它可以盛水")).toBe(
      "杯子可以盛水",
    );
    expect(priorFromInteraction({ state: "needs_clarification" }, "它可以盛水")).toBeNull();
    expect(priorFromInteraction({ state: "clarification_resolved", understanding: null }, "x")).toBe("x");
  });

  test("candidates and status counts", () => {
    const c = summarizeCandidate({
      id: 1,
      status: "proposed",
      kind: "attribute",
      utterance: "杯子有把手",
      parse: { subject: "杯子", predicate: "has", object: "把手" },
      can_confirm: true,
    });
    expect(c).toMatchObject({ id: 1, subject: "杯子", predicate: "has", canConfirm: true });
    expect(countByStatus([{ status: "proposed" }, { status: "accepted" }, { status: "proposed" }])).toEqual({
      all: 3,
      proposed: 2,
      accepted: 1,
      rejected: 0,
    });
  });

  test("statusInfo falls back to the raw value and handles inherited()", () => {
    expect(statusInfo("supported").tone).toBe("good");
    expect(statusInfo("inherited(hypothesis)").label).toBe("继承·假设");
    expect(statusInfo("something_new")).toMatchObject({ label: "something_new", tone: "neutral" });
  });

  test("decision outcomes: only real writes are green; failures are red", () => {
    expect(describeDecision("confirm", { wrote: [{ observation: {} }], reason: "attribute" })).toMatchObject({
      tone: "good",
      status: "accepted",
    });
    // confirmed but nothing written (already known) is not a success highlight
    expect(describeDecision("confirm", { wrote: [], reason: "already known" })).toMatchObject({
      tone: "info",
      message: "知识库中已有该知识，未重复写入",
    });
    expect(describeDecision("reject", { reason: "candidate rejected, nothing written" }).tone).toBe("info");

    const unusable = new ApiError("this shape is not safely storable yet; nothing saved", {
      payload: { ok: false, reason: "this shape is not safely storable yet; nothing saved" },
    });
    expect(describeFailure("confirm", unusable)).toEqual({
      tone: "bad",
      status: "failed",
      message: "录入失败：该句式暂不支持安全写入，未保存任何内容",
    });
    // offline: red message, but the candidate's own status is left alone
    expect(describeFailure("confirm", new ApiError("无法连接服务器", { offline: true })).status).toBe("");
  });

  test("translateReason keeps unknown reasons verbatim", () => {
    expect(translateReason("candidate is accepted, not proposed")).toBe("该候选已处理（已确认），不能重复操作");
    expect(translateReason("no such candidate: 9")).toBe("候选 #9 不存在");
    expect(translateReason("something new")).toBe("something new");
  });
});

import { GREETINGS, pickGreeting, timeGreeting } from "../src/lib/greetings.js";

describe("greetings", () => {
  test("picks from the pool and never repeats the previous one", () => {
    for (let i = 0; i < 200; i++) {
      const prev = GREETINGS[i % GREETINGS.length];
      const g = pickGreeting(prev, { hour: 10 });
      expect(g).not.toBe(prev);
      expect([...GREETINGS, "Good morning"]).toContain(g);
    }
  });

  test("covers both ends of the pool and the time-of-day line", () => {
    expect(pickGreeting("", { random: () => 0, hour: 9 })).toBe(GREETINGS[0]);
    expect(pickGreeting("", { random: () => 0.9999, hour: 9 })).toBe("Good morning");
    expect([timeGreeting(6), timeGreeting(13), timeGreeting(22), timeGreeting(2)]).toEqual([
      "Good morning",
      "Good afternoon",
      "Good evening",
      "Good evening",
    ]);
  });
});
