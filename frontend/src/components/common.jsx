import { useEffect, useRef, useState, useCallback } from "react";
import { statusInfo } from "../lib/labels.js";

export function Icon({ name, className = "", fill = false }) {
  return <i className={className + (fill ? " fill" : "")}>{name}</i>;
}

export function StatusChip({ value, small = true, title }) {
  const s = statusInfo(value);
  return (
    <span className={`status-chip tone-${s.tone}${small ? " small" : ""}`} title={title || s.raw}>
      <i>{s.icon}</i>
      <span>{s.label}</span>
    </span>
  );
}

export function ConceptTag({ match }) {
  if (!match) return null;
  const known = !!match.concept_id;
  return (
    <span className={`concept-tag${known ? "" : " unknown"}`} title={known ? `概念 ID：${match.concept_id}` : "知识库中没有对应概念"}>
      <span>{match.text}</span>
      {known ? <code>{match.concept_id}</code> : <i>question_mark</i>}
    </span>
  );
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fallback for non-secure contexts (http on LAN).
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}

export function IconButton({ icon, label, onClick, className = "", disabled, tooltip = "bottom", children }) {
  return (
    <button
      type="button"
      className={`circle transparent ${className}`}
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
    >
      <i>{icon}</i>
      {children}
      {label && tooltip ? <span className={`tooltip ${tooltip}`}>{label}</span> : null}
    </button>
  );
}

/** Dropdown anchored to its trigger; closes on outside click or Escape. */
export function Dropdown({ trigger, children, align = "left", position = "bottom", className = "" }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e) => ref.current && !ref.current.contains(e.target) && setOpen(false);
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  const close = useCallback(() => setOpen(false), []);
  return (
    <div className={`dropdown ${className}`} ref={ref}>
      {trigger({ open, toggle: () => setOpen((o) => !o) })}
      {open ? (
        <menu className={`dropdown-menu active no-wrap ${position === "top" ? "up" : ""} ${align === "right" ? "align-right" : ""}`}>
          {typeof children === "function" ? children(close) : children}
        </menu>
      ) : null}
    </div>
  );
}

/** Native <dialog> driven by the `open` prop, styled by Beer CSS. */
export function Modal({ open, onClose, children, className = "" }) {
  const ref = useRef(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      className={`modal ${className}`}
      onClose={onClose}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => e.target === ref.current && onClose()}
    >
      {open ? children : null}
    </dialog>
  );
}

export function JsonView({ value }) {
  const text = JSON.stringify(value, null, 2);
  return <pre className="json-view">{text}</pre>;
}

export function Snackbar({ snack, onClose }) {
  useEffect(() => {
    if (!snack) return;
    const t = setTimeout(onClose, snack.timeout ?? 4000);
    return () => clearTimeout(t);
  }, [snack, onClose]);
  if (!snack) return null;
  return (
    <div className={`snackbar active app-snackbar${snack.tone === "error" ? " error" : ""}`} role="status">
      <i>{snack.tone === "error" ? "cancel" : snack.icon || "info"}</i>
      <span className="max">{snack.text}</span>
      {snack.action ? (
        <a
          className="inverse-primary-text"
          onClick={() => {
            snack.action.run();
            onClose();
          }}
        >
          {snack.action.label}
        </a>
      ) : null}
    </div>
  );
}

export function timeAgo(ts, now = Date.now()) {
  const d = Math.max(0, now - ts);
  if (d < 60e3) return "刚刚";
  if (d < 3600e3) return `${Math.floor(d / 60e3)} 分钟前`;
  if (d < 86400e3) return `${Math.floor(d / 3600e3)} 小时前`;
  const dt = new Date(ts);
  return `${dt.getMonth() + 1}月${dt.getDate()}日`;
}
