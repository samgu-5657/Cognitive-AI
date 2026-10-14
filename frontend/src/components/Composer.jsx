import { useEffect, useRef, useState } from "react";
import { MODES, MODE_ORDER } from "../lib/labels.js";

function ContextPanel({ priors, autoContext, onPriors, onAuto }) {
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState(null);
  const add = () => {
    const v = draft.trim();
    if (!v) return;
    onPriors([...priors, v]);
    setDraft("");
  };
  return (
    <div className="context-panel">
      <div className="context-head">
        <i>forum</i>
        <span className="max">
          上下文陈述 <span className="muted">· 按顺序解析，仅保存在本对话</span>
        </span>
        <label className="switch-line" title="理解成功后，把规范化后的句子自动加入上下文">
          <span className="small-text">自动加入</span>
          <label className="switch">
            <input type="checkbox" checked={autoContext} onChange={(e) => onAuto(e.target.checked)} />
            <span />
          </label>
        </label>
        {priors.length ? (
          <button className="transparent small clear-btn" onClick={() => onPriors([])}>
            清空
          </button>
        ) : null}
      </div>
      <div className="prior-chips">
        {priors.map((p, i) =>
          editing === i ? (
            <input
              key={i}
              className="prior-edit"
              autoFocus
              defaultValue={p}
              onBlur={(e) => {
                const v = e.target.value.trim();
                onPriors(v ? priors.map((x, j) => (j === i ? v : x)) : priors.filter((_, j) => j !== i));
                setEditing(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.nativeEvent.isComposing) e.currentTarget.blur();
                if (e.key === "Escape") setEditing(null);
              }}
            />
          ) : (
            <span key={i} className="prior-chip" onDoubleClick={() => setEditing(i)} title="双击编辑">
              <span className="idx">{i + 1}</span>
              <span>{p}</span>
              <button className="x" aria-label="删除" onClick={() => onPriors(priors.filter((_, j) => j !== i))}>
                <i>close</i>
              </button>
            </span>
          ),
        )}
        <span className="prior-add">
          <i>add</i>
          <input
            value={draft}
            placeholder={priors.length ? "添加陈述" : "添加先前陈述，如：杯子是容器"}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.nativeEvent.isComposing) {
                e.preventDefault();
                add();
              }
            }}
            onBlur={add}
          />
        </span>
      </div>
    </div>
  );
}

export default function Composer({
  value,
  onChange,
  mode,
  onMode,
  onSend,
  busy,
  conversation,
  onPriors,
  onAuto,
  inputRef,
}) {
  const local = useRef(null);
  const ref = inputRef || local;
  const m = MODES[mode];

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 220) + "px";
  }, [value, ref]);

  const canSend = value.trim().length > 0 && !busy;
  const submit = () => canSend && onSend(value.trim());

  return (
    <div className="composer-wrap">
      <div className={`composer ${busy ? "busy" : ""}`}>
        {mode === "context" ? (
          <ContextPanel
            priors={conversation?.priors || []}
            autoContext={conversation?.autoContext ?? true}
            onPriors={onPriors}
            onAuto={onAuto}
          />
        ) : null}
        <textarea
          ref={ref}
          rows={1}
          value={value}
          placeholder={m.placeholder}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          aria-label="输入"
        />
        <div className="composer-bar">
          <div className="mode-chips" role="tablist" aria-label="模式">
            {MODE_ORDER.map((id, i) => {
              const mm = MODES[id];
              const on = id === mode;
              return (
                <button
                  key={id}
                  role="tab"
                  aria-selected={on}
                  className={`mode-chip ${on ? "on" : ""}`}
                  onClick={() => onMode(id)}
                  title={`${mm.hint}（Alt+${i + 1}）`}
                >
                  <i>{mm.icon}</i>
                  <span>{mm.short}</span>
                </button>
              );
            })}
          </div>
          <span className="max" />
          <button
            className={`send-btn circle ${canSend ? "ready" : ""}`}
            disabled={!canSend}
            onClick={submit}
            aria-label="发送"
          >
            {busy ? <progress className="circle small" /> : <i>arrow_upward</i>}
          </button>
        </div>
      </div>
      <p className="disclaimer">
        {m.hint}。回答只依据本地知识库的符号规则与记录，可能不完整。
      </p>
    </div>
  );
}
