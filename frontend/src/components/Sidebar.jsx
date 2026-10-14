import { useState } from "react";
import { Dropdown, IconButton, timeAgo } from "./common.jsx";

function ConvItem({ conv, active, onSelect, onRename, onDelete }) {
  const [editing, setEditing] = useState(false);
  if (editing)
    return (
      <li className="conv-item editing">
        <input
          autoFocus
          defaultValue={conv.title}
          onBlur={(e) => {
            if (e.target.value.trim()) onRename(conv.id, e.target.value);
            setEditing(false);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.nativeEvent.isComposing) e.currentTarget.blur();
            if (e.key === "Escape") setEditing(false);
          }}
        />
      </li>
    );
  return (
    <li className={`conv-item ${active ? "active" : ""}`}>
      <button className="conv-main" onClick={() => onSelect(conv.id)} title={`${conv.title} · ${timeAgo(conv.updatedAt)}`}>
        <span className="conv-title">{conv.title}</span>
      </button>
      <Dropdown
        align="right"
        trigger={({ toggle }) => (
          <button className="conv-more circle transparent small" onClick={toggle} aria-label="更多">
            <i>more_vert</i>
          </button>
        )}
      >
        {(close) => (
          <>
            <li
              onClick={() => {
                close();
                setEditing(true);
              }}
            >
              <i>edit</i>
              <span>重命名</span>
            </li>
            <li
              onClick={() => {
                close();
                onDelete(conv.id);
              }}
            >
              <i>delete</i>
              <span>删除</span>
            </li>
          </>
        )}
      </Dropdown>
    </li>
  );
}

export default function Sidebar({
  open,
  view,
  onView,
  conversations,
  activeId,
  onNew,
  onSelect,
  onRename,
  onDelete,
  status,
  onRefreshStatus,
  onSettings,
  onClose,
  onToggle,
}) {
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();
  const list = q
    ? conversations.filter(
        (c) => c.title.toLowerCase().includes(q) || c.messages.some((m) => m.role === "user" && m.text.toLowerCase().includes(q)),
      )
    : conversations;
  return (
    <>
      <div className={`sidebar-scrim ${open ? "show" : ""}`} onClick={onClose} />
      <div className={`sidebar ${open ? "open" : "closed"}`} role="navigation" aria-label="侧边栏">
        <div className="sidebar-head">
          <IconButton icon="menu" label={open ? "收起侧边栏" : "展开侧边栏"} tooltip="right" onClick={onToggle} />
        </div>
        <div className="sidebar-top">
          <button className="new-chat" onClick={onNew}>
            <i>edit_square</i>
            <span>发起新对话</span>
          </button>
          <button className={`side-link ${view === "review" ? "active" : ""}`} onClick={() => onView("review")}>
            <i>fact_check</i>
            <span className="max">知识审核</span>
            {status?.pending ? <span className="pill-badge">{status.pending}</span> : null}
          </button>
        </div>

        <div className="sidebar-section">
          <div className="section-label">
            <span className="max">最近</span>
          </div>
          {conversations.length > 4 ? (
            <div className="side-search">
              <i>search</i>
              <input value={query} placeholder="搜索对话" onChange={(e) => setQuery(e.target.value)} />
            </div>
          ) : null}
          <ul className="conv-list">
            {list.map((c) => (
              <ConvItem
                key={c.id}
                conv={c}
                active={view === "chat" && c.id === activeId}
                onSelect={onSelect}
                onRename={onRename}
                onDelete={onDelete}
              />
            ))}
            {!list.length ? <li className="muted small-text empty-note">{q ? "没有匹配的对话" : "还没有对话"}</li> : null}
          </ul>
        </div>

        <div className="sidebar-bottom">
          <button className="status-card" onClick={onRefreshStatus} title="点击刷新">
            <span className={`dot ${status?.online ? "on" : status?.online === false ? "off" : ""}`} />
            {status?.online ? (
              <span className="status-grid">
                <span>
                  <b>{status.concepts}</b> 概念
                </span>
                <span>
                  <b>{status.rules}</b> 规则
                </span>
                <span>
                  <b>v{status.version ?? "—"}</b> 版本
                </span>
              </span>
            ) : (
              <span className="max">{status?.online === false ? "后端未连接" : "正在连接…"}</span>
            )}
          </button>
          <button className="side-link" onClick={onSettings}>
            <i>settings</i>
            <span>设置</span>
          </button>
        </div>
      </div>
    </>
  );
}
