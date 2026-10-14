import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { ui } from "beercss";
import { api as defaultApi } from "./lib/api.js";
import { MODES, MODE_ORDER } from "./lib/labels.js";
import { reducer, loadState, saveState, sortedConversations, uid, newConversation } from "./lib/conversations.js";
import { loadPrefs, savePrefs, safeStorage } from "./lib/prefs.js";
import { pickGreeting } from "./lib/greetings.js";
import { priorFromInteraction, describeDecision, describeFailure } from "./lib/summarize.js";
import Sidebar from "./components/Sidebar.jsx";
import Thread, { copyAndNotify } from "./components/Thread.jsx";
import Composer from "./components/Composer.jsx";
import Welcome from "./components/Welcome.jsx";
import ReviewPage from "./components/ReviewPage.jsx";
import SettingsDialog from "./components/SettingsDialog.jsx";
import { Modal, JsonView, Snackbar, IconButton, Dropdown } from "./components/common.jsx";

const NARROW = "(max-width: 900px)";
const isNarrow = () => typeof matchMedia === "function" && matchMedia(NARROW).matches;

export default function App({ api = defaultApi }) {
  const storage = useMemo(() => safeStorage(), []);
  const [state, dispatch] = useReducer(reducer, undefined, () => loadState(storage));
  const [prefs, setPrefs] = useState(() => loadPrefs(storage));
  const [view, setView] = useState("chat");
  const [draft, setDraft] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(() => !isNarrow() && loadPrefs(storage).sidebar);
  const [status, setStatus] = useState({ online: null });
  const [snack, setSnack] = useState(null);
  const [jsonMsg, setJsonMsg] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [greeting, setGreeting] = useState(() => pickGreeting());
  const inputRef = useRef(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  const mode = prefs.mode in MODES ? prefs.mode : "ask";
  const conversations = useMemo(() => sortedConversations(state), [state]);
  const active = state.conversations.find((c) => c.id === state.activeId) || null;
  const busy = !!active?.messages.some((m) => m.status === "loading");

  const notify = useCallback((s) => setSnack({ ...s, key: Date.now() }), []);
  const closeSnack = useCallback(() => setSnack(null), []);

  // ---- persistence -------------------------------------------------------
  useEffect(() => {
    const t = setTimeout(() => saveState(storage, state), 250);
    return () => clearTimeout(t);
  }, [state, storage]);

  const updatePrefs = useCallback(
    (patch) =>
      setPrefs((p) => {
        const next = { ...p, ...patch };
        savePrefs(next, storage);
        return next;
      }),
    [storage],
  );

  // ---- theme -------------------------------------------------------------
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    const apply = () => {
      try {
        ui("mode", prefs.themeMode);
        setIsDark(ui("mode") === "dark");
      } catch {
        /* ignore */
      }
    };
    apply();
    if (prefs.themeMode !== "auto" || typeof matchMedia !== "function") return;
    const mq = matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [prefs.themeMode]);
  useEffect(() => {
    Promise.resolve()
      .then(() => ui("theme", prefs.seed))
      .catch(() => {});
  }, [prefs.seed]);

  // ---- backend status ----------------------------------------------------
  const refreshStatus = useCallback(async () => {
    try {
      const s = await api.status();
      setStatus({ online: true, ...s });
    } catch (e) {
      setStatus((prev) => ({ ...prev, online: e.offline || e.status >= 500 || !e.status ? false : prev.online }));
    }
  }, [api]);

  useEffect(() => {
    refreshStatus();
    const t = setInterval(refreshStatus, 20000);
    const onFocus = () => refreshStatus();
    window.addEventListener("focus", onFocus);
    return () => {
      clearInterval(t);
      window.removeEventListener("focus", onFocus);
    };
  }, [refreshStatus]);

  // ---- helpers -----------------------------------------------------------
  const patch = useCallback((convId, msgId, p) => dispatch({ type: "patchMessage", convId, msgId, patch: p }), []);
  const patchData = useCallback(
    (convId, msgId, dataPatch) => dispatch({ type: "patchData", convId, msgId, patch: dataPatch }),
    [],
  );
  const getConv = (id) => stateRef.current.conversations.find((c) => c.id === id);
  const addPrior = (convId, prior) => {
    const conv = getConv(convId);
    if (!conv || !prior) return false;
    if (conv.priors[conv.priors.length - 1] === prior) return false;
    dispatch({ type: "setPriors", convId, priors: [...conv.priors, prior] });
    return true;
  };

  // ---- the main action ---------------------------------------------------
  const run = useCallback(
    async (text, runMode, opts = {}) => {
      const cur = stateRef.current;
      const existing = cur.conversations.find((c) => c.id === cur.activeId);
      const convId = existing ? existing.id : uid("c");
      if (opts.priors) {
        if (!getConv(convId)) dispatch({ type: "restore", conversation: { ...newConversation(), id: convId } });
        dispatch({ type: "setPriors", convId, priors: opts.priors });
      }
      const priors = opts.priors || getConv(convId)?.priors || [];
      const user = { id: uid("m"), role: "user", mode: runMode, text, ts: Date.now() };
      const asst = { id: uid("m"), role: "assistant", mode: runMode, status: "loading", replyTo: user.id, ts: Date.now() };
      dispatch({ type: "send", convId, user, assistant: asst });
      setView("chat");

      try {
        let data;
        if (runMode === "ask") data = await api.ask(text);
        else if (runMode === "analyze") data = await api.analyze(text);
        else if (runMode === "teach") data = await api.propose(text);
        else {
          const [inter, ctx] = await Promise.all([api.interact(text, priors), api.context(text, priors)]);
          data = { text, priors, interact: inter.result, report: ctx.report, mentions: ctx.context };
          const conv = getConv(convId);
          const prior = priorFromInteraction(inter.result, text);
          if (prior && (conv?.autoContext ?? true) && addPrior(convId, prior)) data.addedPrior = prior;
        }
        const { ok, ...rest } = data;
        patch(convId, asst.id, { status: "done", data: rest });
        if (runMode === "teach") refreshStatus();
      } catch (e) {
        patch(convId, asst.id, { status: "error", error: e.message, offline: !!e.offline });
        if (e.offline) setStatus((s) => ({ ...s, online: false }));
      }
    },
    [api, patch, refreshStatus],
  );

  const send = (text) => {
    if (busy) return;
    setDraft("");
    run(text, mode);
  };

  // ---- message handlers --------------------------------------------------
  const handlers = {
    copy: (text) => copyAndNotify(text, notify),
    edit: (msg) => {
      setDraft(msg.text);
      updatePrefs({ mode: msg.mode });
      requestAnimationFrame(() => inputRef.current?.focus());
    },
    rerun: (userMsg, m) => !busy && run(userMsg.text, m),
    showJson: (msg) => setJsonMsg(msg),
    choose: async (msg, option) => {
      const convId = state.activeId;
      const d = msg.data;
      patchData(convId, msg.id, { resolving: true });
      try {
        const res = await api.answer(d.interact.request, option.option_id, d.priors, d.text);
        const extra = { resolving: false, resolution: res.result };
        const prior = priorFromInteraction(res.result, d.text);
        if (prior && (getConv(convId)?.autoContext ?? true) && addPrior(convId, prior)) extra.addedPrior = prior;
        patchData(convId, msg.id, extra);
      } catch (e) {
        patchData(convId, msg.id, { resolving: false });
        notify({ text: e.message, tone: "error" });
      }
    },
    cancelClarify: (msg) => patchData(state.activeId, msg.id, { cancelled: true }),
    decide: async (msg, id, how) => {
      const convId = state.activeId;
      patchData(convId, msg.id, { deciding: true });
      try {
        const res = await api[how](id);
        markDecided(id, how, res);
        const d = describeDecision(how, res);
        notify({ text: `候选 #${id}：${d.message}`, icon: d.tone === "good" ? "check_circle" : "info" });
      } catch (e) {
        const d = describeFailure(how, e);
        patchData(convId, msg.id, { decision: d });
        notify({ text: `候选 #${id}：${d.message}`, tone: "error" });
      } finally {
        patchData(convId, msg.id, { deciding: false });
        refreshStatus();
      }
    },
  };

  // Reflect a decision on every chat message that shows the same candidate.
  const markDecided = useCallback(
    (id, how, res) => {
      for (const conv of stateRef.current.conversations)
        for (const m of conv.messages)
          if (m.mode === "teach" && m.data?.candidate?.id === id)
            patchData(conv.id, m.id, {
              decision: describeDecision(how, res),
            });
    },
    [patchData],
  );

  // ---- navigation --------------------------------------------------------
  const closeOnNarrow = () => isNarrow() && setSidebarOpen(false);
  const newChat = useCallback(() => {
    dispatch({ type: "new" });
    setGreeting((g) => pickGreeting(g));
    setView("chat");
    setDraft("");
    closeOnNarrow();
    requestAnimationFrame(() => inputRef.current?.focus());
  }, []);
  const toggleSidebar = () =>
    setSidebarOpen((o) => {
      if (!isNarrow()) updatePrefs({ sidebar: !o });
      return !o;
    });

  // ---- keyboard shortcuts ------------------------------------------------
  useEffect(() => {
    const onKey = (e) => {
      if (e.altKey && !e.ctrlKey && !e.metaKey && /^Digit[1-4]$/.test(e.code)) {
        e.preventDefault();
        updatePrefs({ mode: MODE_ORDER[Number(e.code.slice(5)) - 1] });
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") {
        e.preventDefault();
        newChat();
        return;
      }
      const tag = document.activeElement?.tagName;
      if (e.key === "/" && tag !== "INPUT" && tag !== "TEXTAREA" && !settingsOpen && !jsonMsg) {
        e.preventDefault();
        setView("chat");
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [newChat, updatePrefs, settingsOpen, jsonMsg]);

  const pickSuggestion = (s) => {
    updatePrefs({ mode: s.mode });
    run(s.text, s.mode, { priors: s.priors });
  };

  const empty = !active || active.messages.length === 0;

  return (
    <div className={`app ${sidebarOpen ? "with-sidebar" : ""}`}>
      <Sidebar
        open={sidebarOpen}
        view={view}
        onView={(v) => {
          setView(v);
          closeOnNarrow();
        }}
        conversations={conversations}
        activeId={state.activeId}
        onNew={newChat}
        onSelect={(id) => {
          dispatch({ type: "select", id });
          setView("chat");
          closeOnNarrow();
        }}
        onRename={(id, title) => dispatch({ type: "rename", id, title })}
        onDelete={(id) => {
          const conv = getConv(id);
          dispatch({ type: "delete", id });
          notify({
            text: `已删除「${conv?.title || "对话"}」`,
            icon: "delete",
            action: conv ? { label: "撤销", run: () => dispatch({ type: "restore", conversation: conv }) } : null,
          });
        }}
        status={status}
        onRefreshStatus={refreshStatus}
        onSettings={() => setSettingsOpen(true)}
        onClose={() => setSidebarOpen(false)}
        onToggle={toggleSidebar}
      />

      <div className="main-pane">
        <div className="topbar">
          <IconButton icon="menu" label="展开侧边栏" onClick={toggleSidebar} className="menu-top" />
          <div className="brand">
            <span className="brand-name">Cognitive AI</span>
            {view === "chat" ? (
              <Dropdown
                trigger={({ toggle, open }) => (
                  <button className="mode-picker" onClick={toggle}>
                    <span>{MODES[mode].label}</span>
                    <i>{open ? "arrow_drop_up" : "arrow_drop_down"}</i>
                  </button>
                )}
              >
                {(close) =>
                  MODE_ORDER.map((id, i) => (
                    <li
                      key={id}
                      className={id === mode ? "active-item" : ""}
                      onClick={() => {
                        updatePrefs({ mode: id });
                        close();
                        inputRef.current?.focus();
                      }}
                    >
                      <i>{MODES[id].icon}</i>
                      <div className="max">
                        <div>{MODES[id].label}</div>
                        <div className="small-text muted">{MODES[id].hint}</div>
                      </div>
                      <kbd>Alt+{i + 1}</kbd>
                    </li>
                  ))
                }
              </Dropdown>
            ) : (
              <span className="crumb">知识审核</span>
            )}
          </div>
          <span className="max" />
          {status.online === false ? (
            <span className="offline-chip" title="无法连接 Python 后端">
              <i>cloud_off</i>
              <span>后端离线</span>
            </span>
          ) : null}
          <IconButton
            icon={isDark ? "light_mode" : "dark_mode"}
            label={isDark ? "切换到浅色" : "切换到深色"}
            onClick={() => updatePrefs({ themeMode: isDark ? "light" : "dark" })}
          />
          <IconButton icon="edit_square" label="新对话" onClick={newChat} className="new-chat-top" />
        </div>

        {view === "review" ? (
          <div className="scroll-area">
            <ReviewPage
              api={api}
              notify={notify}
              onChanged={refreshStatus}
              onDecided={(id, how, res) => markDecided(id, how, res)}
            />
          </div>
        ) : (
          <>
            <div className="scroll-area">
              {empty ? <Welcome greeting={greeting} status={status} onPick={pickSuggestion} /> : <Thread conversation={active} handlers={handlers} />}
            </div>
            <Composer
              value={draft}
              onChange={setDraft}
              mode={mode}
              onMode={(m) => updatePrefs({ mode: m })}
              onSend={send}
              busy={busy}
              conversation={active}
              inputRef={inputRef}
              onPriors={(priors) => {
                let id = active?.id;
                if (!id) {
                  const conv = newConversation();
                  dispatch({ type: "restore", conversation: conv });
                  id = conv.id;
                }
                dispatch({ type: "setPriors", convId: id, priors });
              }}
              onAuto={(value) => active && dispatch({ type: "setAutoContext", convId: active.id, value })}
            />
          </>
        )}
      </div>

      <Modal open={!!jsonMsg} onClose={() => setJsonMsg(null)} className="json-dialog">
        <div className="dialog-head">
          <h6 className="max">原始响应 · {MODES[jsonMsg?.mode]?.label}</h6>
          <IconButton
            icon="content_copy"
            label="复制 JSON"
            onClick={() => handlers.copy(JSON.stringify(jsonMsg?.data, null, 2))}
          />
          <IconButton icon="close" label="关闭" onClick={() => setJsonMsg(null)} />
        </div>
        <JsonView value={jsonMsg?.data} />
      </Modal>

      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        prefs={prefs}
        onPrefs={updatePrefs}
        conversationCount={state.conversations.length}
        onClearAll={() => {
          dispatch({ type: "clearAll" });
          notify({ text: "已清除全部对话" });
        }}
      />

      <Snackbar key={snack?.key} snack={snack} onClose={closeSnack} />
    </div>
  );
}
