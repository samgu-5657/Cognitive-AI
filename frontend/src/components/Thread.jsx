import { useEffect, useRef } from "react";
import { MODES, MODE_ORDER } from "../lib/labels.js";
import { IconButton, Dropdown, copyText } from "./common.jsx";
import { AskResult, AnalyzeResult, ContextResult, TeachResult } from "./results.jsx";
import { plainTextOf } from "../lib/summarize.js";

function Spark({ busy }) {
  return (
    <div className={`spark ${busy ? "busy" : ""}`} aria-hidden="true">
      <svg viewBox="0 0 24 24">
        <path d="M12 2c.6 5.4 4.6 9.4 10 10-5.4.6-9.4 4.6-10 10-.6-5.4-4.6-9.4-10-10 5.4-.6 9.4-4.6 10-10z" />
      </svg>
    </div>
  );
}

const LOADING_TEXT = {
  ask: "正在知识库中查找证据…",
  analyze: "正在拆分结构并逐项查证…",
  context: "正在结合上下文解析指代…",
  teach: "正在生成知识候选…",
};

function UserMessage({ msg, onEdit, onRerun, onCopy }) {
  const mode = MODES[msg.mode] || MODES.ask;
  return (
    <div className="msg user">
      <div className="user-actions">
        <IconButton icon="content_copy" label="复制" onClick={() => onCopy(msg.text)} />
        <IconButton icon="edit" label="编辑后重新发送" onClick={() => onEdit(msg)} />
      </div>
      <div className="user-bubble">
        <span className="mode-tag">
          <i>{mode.icon}</i>
          {mode.short}
        </span>
        <div className="user-text">{msg.text}</div>
      </div>
    </div>
  );
}

function AssistantMessage({ msg, user, handlers }) {
  const busy = msg.status === "loading";
  const mode = msg.mode;
  let body;
  if (busy) body = <p className="loading-text shimmer">{LOADING_TEXT[mode] || "处理中…"}</p>;
  else if (msg.status === "error")
    body = (
      <div className="error-box">
        <i>error</i>
        <div className="max">
          <div>{msg.error}</div>
          {msg.offline ? <div className="small-text">请先运行 <code>python -m cognitive_ai.cli ui</code></div> : null}
        </div>
        <button className="border small" onClick={() => handlers.rerun(user, mode)}>
          <i>refresh</i>
          <span>重试</span>
        </button>
      </div>
    );
  else if (mode === "ask") body = <AskResult data={msg.data} />;
  else if (mode === "analyze") body = <AnalyzeResult data={msg.data} />;
  else if (mode === "context")
    body = (
      <ContextResult
        data={msg.data}
        onChoose={(opt) => handlers.choose(msg, opt)}
        onCancel={() => handlers.cancelClarify(msg)}
      />
    );
  else if (mode === "teach") body = <TeachResult data={msg.data} onDecide={(id, how) => handlers.decide(msg, id, how)} />;

  return (
    <div className="msg assistant">
      <Spark busy={busy} />
      <div className="assistant-body">
        {body}
        {!busy && msg.status === "done" ? (
          <div className="assistant-actions">
            <IconButton
              icon="content_copy"
              label="复制回答"
              tooltip="top"
              onClick={() => handlers.copy(plainTextOf(mode, msg.data) || JSON.stringify(msg.data))}
            />
            <IconButton icon="data_object" label="查看原始 JSON" tooltip="top" onClick={() => handlers.showJson(msg)} />
            {user ? (
              <IconButton icon="refresh" label="重新运行" tooltip="top" onClick={() => handlers.rerun(user, mode)} />
            ) : null}
            {user ? (
              <Dropdown
                position="top"
                trigger={({ toggle }) => <IconButton icon="swap_horiz" label="换个模式运行" tooltip="top" onClick={toggle} />}
              >
                {(close) =>
                  MODE_ORDER.filter((m) => m !== mode).map((m) => (
                    <li
                      key={m}
                      onClick={() => {
                        close();
                        handlers.rerun(user, m);
                      }}
                    >
                      <i>{MODES[m].icon}</i>
                      <span>用「{MODES[m].label}」运行</span>
                    </li>
                  ))
                }
              </Dropdown>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default function Thread({ conversation, handlers }) {
  const endRef = useRef(null);
  const count = conversation?.messages.length ?? 0;
  const last = conversation?.messages[count - 1];
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [count, last?.status]);

  const msgs = conversation?.messages || [];
  const byId = Object.fromEntries(msgs.map((m) => [m.id, m]));
  return (
    <div className="thread">
      {msgs.map((m) =>
        m.role === "user" ? (
          <UserMessage key={m.id} msg={m} onEdit={handlers.edit} onCopy={handlers.copy} />
        ) : (
          <AssistantMessage key={m.id} msg={m} user={byId[m.replyTo]} handlers={handlers} />
        ),
      )}
      <div ref={endRef} className="thread-end" />
    </div>
  );
}

export function copyAndNotify(text, notify) {
  copyText(text).then((ok) => notify(ok ? { text: "已复制", icon: "content_copy" } : { text: "复制失败", tone: "error" }));
}
