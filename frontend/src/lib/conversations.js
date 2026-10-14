// Client-side conversation state. Everything here lives only in the browser
// (localStorage); the backend is stateless and never sees this history.

export const STORAGE_KEY = "cognitive-ai.conversations.v1";
export const MAX_CONVERSATIONS = 50;

let seq = 0;
export function uid(prefix = "id") {
  seq = (seq + 1) % 1e6;
  return `${prefix}-${Date.now().toString(36)}-${seq.toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

export function titleFrom(text) {
  const t = String(text || "").replace(/\s+/g, " ").trim();
  if (!t) return "新对话";
  return t.length > 24 ? t.slice(0, 24) + "…" : t;
}

export function newConversation(now = Date.now()) {
  return {
    id: uid("c"),
    title: "新对话",
    titleLocked: false,
    createdAt: now,
    updatedAt: now,
    priors: [],
    autoContext: true,
    messages: [],
  };
}

export function initialState() {
  return { conversations: [], activeId: null };
}

function mapConv(state, id, fn) {
  return {
    ...state,
    conversations: state.conversations.map((c) => (c.id === id ? fn(c) : c)),
  };
}

export function reducer(state, action) {
  switch (action.type) {
    case "new": {
      const active = state.conversations.find((c) => c.id === state.activeId);
      // Reuse an untouched empty conversation instead of piling up blanks.
      if (active && active.messages.length === 0 && !active.priors.length) return state;
      const conv = action.conversation || newConversation();
      return {
        conversations: [conv, ...state.conversations].slice(0, MAX_CONVERSATIONS),
        activeId: conv.id,
      };
    }
    case "restore":
      if (state.conversations.some((c) => c.id === action.conversation.id)) return state;
      return {
        conversations: [action.conversation, ...state.conversations].slice(0, MAX_CONVERSATIONS),
        activeId: action.conversation.id,
      };
    case "select":
      return state.conversations.some((c) => c.id === action.id) ? { ...state, activeId: action.id } : state;
    case "delete": {
      const conversations = state.conversations.filter((c) => c.id !== action.id);
      const activeId = state.activeId === action.id ? conversations[0]?.id ?? null : state.activeId;
      return { conversations, activeId };
    }
    case "rename":
      return mapConv(state, action.id, (c) => ({
        ...c,
        title: titleFrom(action.title),
        titleLocked: true,
      }));
    case "clearAll":
      return initialState();
    case "send": {
      // The caller picks the conversation id; create it on first send.
      let s = state;
      if (!s.conversations.some((c) => c.id === action.convId)) {
        const conv = { ...newConversation(), id: action.convId };
        s = { conversations: [conv, ...s.conversations].slice(0, MAX_CONVERSATIONS), activeId: conv.id };
      }
      const now = action.now ?? Date.now();
      return mapConv(s, action.convId, (c) => ({
        ...c,
        title: c.titleLocked || c.messages.length ? c.title : titleFrom(action.user.text),
        updatedAt: now,
        messages: [...c.messages, action.user, action.assistant],
      }));
    }
    case "patchMessage":
      return mapConv(state, action.convId, (c) => ({
        ...c,
        updatedAt: action.now ?? Date.now(),
        messages: c.messages.map((m) => (m.id === action.msgId ? { ...m, ...action.patch } : m)),
      }));
    case "patchData":
      return mapConv(state, action.convId, (c) => ({
        ...c,
        messages: c.messages.map((m) => (m.id === action.msgId ? { ...m, data: { ...(m.data || {}), ...action.patch } } : m)),
      }));
    case "setPriors":
      return mapConv(state, action.convId, (c) => ({
        ...c,
        priors: action.priors.filter((p) => typeof p === "string" && p.trim()),
      }));
    case "setAutoContext":
      return mapConv(state, action.convId, (c) => ({ ...c, autoContext: !!action.value }));
    default:
      return state;
  }
}

export function sortedConversations(state) {
  return [...state.conversations].sort((a, b) => b.updatedAt - a.updatedAt);
}

// A request that was in flight when the page closed can never finish.
function settle(conv) {
  return {
    ...conv,
    messages: conv.messages.map((m) =>
      m.status === "loading" ? { ...m, status: "error", error: "页面关闭时请求未完成，请重试" } : m,
    ),
  };
}

export function loadState(storage) {
  try {
    const raw = storage?.getItem(STORAGE_KEY);
    if (!raw) return initialState();
    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.conversations)) return initialState();
    const conversations = parsed.conversations
      .filter((c) => c && typeof c.id === "string" && Array.isArray(c.messages))
      .map((c) => settle({ ...newConversation(c.createdAt), ...c, priors: Array.isArray(c.priors) ? c.priors : [] }));
    const activeId = conversations.some((c) => c.id === parsed.activeId) ? parsed.activeId : conversations[0]?.id ?? null;
    return { conversations, activeId };
  } catch {
    return initialState();
  }
}

export function saveState(storage, state) {
  if (!storage) return false;
  let convs = sortedConversations(state);
  // On quota errors drop the oldest conversations until it fits.
  while (true) {
    try {
      storage.setItem(STORAGE_KEY, JSON.stringify({ conversations: convs, activeId: state.activeId }));
      return true;
    } catch {
      if (convs.length <= 1) return false;
      convs = convs.slice(0, Math.ceil(convs.length / 2));
    }
  }
}
