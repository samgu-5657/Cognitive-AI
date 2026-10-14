import { useState } from "react";
import { StatusChip, ConceptTag } from "./common.jsx";
import {
  summarizeAsk,
  summarizeUnderstanding,
  summarizeContext,
  contextMentions,
  summarizeCandidate,
  translateReason,
} from "../lib/summarize.js";
import { EXPR_KIND, ROLE, PREDICATE, QKIND, CANDIDATE_KIND, statusInfo, label } from "../lib/labels.js";

function Section({ icon, title, children, defaultOpen = false, count }) {
  return (
    <details className="result-section" open={defaultOpen}>
      <summary>
        <i>{icon}</i>
        <span className="max">{title}</span>
        {count != null ? <span className="count">{count}</span> : null}
        <i className="chev">expand_more</i>
      </summary>
      <div className="section-body">{children}</div>
    </details>
  );
}

function Evidence({ items }) {
  if (!items?.length) return <p className="muted small-text">没有找到证据记录。</p>;
  return (
    <ul className="evidence-list">
      {items.map((e, i) => (
        <li key={i}>
          <div className="row-line">
            <span className="kv-key">{e.kind}</span>
            <code>{e.ref}</code>
            <span className="max" />
            <StatusChip value={e.status} />
          </div>
          {e.detail ? <code className="detail-code">{e.detail}</code> : null}
        </li>
      ))}
    </ul>
  );
}

function Query({ q }) {
  const { op, ...rest } = q || {};
  return (
    <div className="query-line">
      <code className="op">{op}</code>
      {Object.entries(rest).map(([k, v]) => (
        <span key={k} className="kv">
          <span className="kv-key">{k}</span>
          <span>{v === null ? "null" : String(v)}</span>
        </span>
      ))}
    </div>
  );
}

/* ----------------------------- 知识问答 ----------------------------- */

export function AskResult({ data }) {
  const s = summarizeAsk(data);
  return (
    <div className="result ask-result">
      <p className="answer-text">{s.answer || "（没有生成回答）"}</p>
      <div className="chip-row">
        <StatusChip value={s.verdict} small={false} />
        {s.qkind ? <span className="soft-chip">{label(QKIND, s.qkind)}</span> : null}
        {s.limited ? <span className="soft-chip warn">结果受限</span> : null}
        {s.ambiguity ? <span className="soft-chip warn">歧义：{s.ambiguity}</span> : null}
      </div>
      <Section icon="psychology" title="查证过程" count={s.evidence.length}>
        <div className="grid-kv">
          <span className="kv-key">主语</span>
          <span>{s.subject ? <ConceptTag match={s.subject} /> : "—"}</span>
          <span className="kv-key">宾语</span>
          <span>{s.object ? <ConceptTag match={s.object} /> : "—"}</span>
          {s.unknownSpans.length ? (
            <>
              <span className="kv-key">未识别片段</span>
              <span>{s.unknownSpans.join("、")}</span>
            </>
          ) : null}
          <span className="kv-key">结论说明</span>
          <span className="mono-soft">{s.detail || "—"}</span>
        </div>
        {s.queries.length ? (
          <>
            <h6 className="sub-title">查询</h6>
            {s.queries.map((q, i) => (
              <Query key={i} q={q} />
            ))}
          </>
        ) : null}
        <h6 className="sub-title">证据</h6>
        <Evidence items={s.evidence} />
        {s.constructions.length ? (
          <p className="muted small-text">
            生成句式：{s.constructions.join("、")} · {s.roundtrippable ? "可回译" : "不可回译"}
          </p>
        ) : null}
      </Section>
    </div>
  );
}

/* ----------------------------- 语义分析 ----------------------------- */

function TreeNode({ node, depth = 0 }) {
  const [open, setOpen] = useState(depth < 2);
  const hasKids = node.children.length > 0;
  const tone = statusInfo(node.status).tone;
  return (
    <li className={`tree-node tone-border-${tone}`}>
      <div className="tree-head" onClick={() => setOpen((o) => !o)}>
        {hasKids ? <i className={`chev ${open ? "open" : ""}`}>chevron_right</i> : <i className="dot">circle</i>}
        {node.role ? <span className="role-tag">{label(ROLE, node.role)}</span> : null}
        <span className="kind-tag">{label(EXPR_KIND, node.kind)}</span>
        {node.connective ? <span className="soft-chip">“{node.connective}”</span> : null}
        <span className="tree-text max">{node.text}</span>
        <StatusChip value={node.status} />
      </div>
      {open ? (
        <div className="tree-body">
          {node.atom ? (
            <div className="atom-line">
              <span className="atom-part">{node.atom.subject || "?"}</span>
              <span className="atom-pred">
                {node.atom.negated ? "不" : ""}
                {label(PREDICATE, node.atom.predicate)}
              </span>
              <span className="atom-part">{node.atom.object || "?"}</span>
              {node.subject ? <ConceptTag match={node.subject} /> : null}
              {node.object ? <ConceptTag match={node.object} /> : null}
            </div>
          ) : null}
          {node.detail ? <div className="mono-soft small-text">{node.detail}</div> : null}
          {node.evidence.length ? <Evidence items={node.evidence} /> : null}
          {hasKids ? (
            <ul className="tree">
              {node.children.map((c, i) => (
                <TreeNode key={i} node={c} depth={depth + 1} />
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}

export function UnderstandingView({ report, compact = false }) {
  const u = summarizeUnderstanding(report);
  if (!u) return null;
  return (
    <div className="understanding">
      <div className="chip-row">
        <StatusChip value={u.tree?.status} small={false} />
        <span className={`soft-chip ${u.complete ? "good" : "warn"}`}>{u.complete ? "结构完整" : "结构不完整"}</span>
        <span className="soft-chip">原子 {u.atomCount}</span>
        {u.counts.map(([k, n]) => (
          <span key={k} className={`soft-chip tone-${statusInfo(k).tone}`}>
            {statusInfo(k).label} × {n}
          </span>
        ))}
      </div>
      {u.tree ? (
        <ul className={`tree root ${compact ? "compact" : ""}`}>
          <TreeNode node={u.tree} />
        </ul>
      ) : null}
      {u.diagnostics.length ? (
        <ul className="diag-list">
          {u.diagnostics.map((d, i) => (
            <li key={i}>{d}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

export function AnalyzeResult({ data }) {
  return (
    <div className="result">
      <UnderstandingView report={data?.report} />
    </div>
  );
}

/* ---------------------------- 上下文理解 ---------------------------- */

function Resolutions({ items }) {
  if (!items?.length) return <p className="muted small-text">句中没有需要解析的指代。</p>;
  return (
    <ul className="resolution-list">
      {items.map((r, i) => (
        <li key={i}>
          <span className="mention">“{r.mention}”</span>
          <i>arrow_forward</i>
          <span className="target">{r.target || (r.candidates.length ? r.candidates.join(" / ") : "—")}</span>
          <StatusChip value={r.status} />
          {r.reason ? <span className="muted small-text max">{r.reason}</span> : null}
        </li>
      ))}
    </ul>
  );
}

function Clarification({ request, busy, onChoose, onCancel }) {
  return (
    <div className="clarify-card">
      <div className="clarify-q">
        <i>contact_support</i>
        <span>{request.question}</span>
      </div>
      <div className="chip-row">
        {(request.options || []).map((o) => (
          <button
            key={o.option_id}
            className="chip option-chip"
            disabled={!o.selectable || busy}
            title={o.selectable ? o.ref?.ref_id : `不可选：${o.reason}`}
            onClick={() => onChoose(o)}
          >
            <i>{o.selectable ? "radio_button_unchecked" : "block"}</i>
            <span>{o.display_name}</span>
          </button>
        ))}
        <button className="chip border transparent" disabled={busy} onClick={onCancel}>
          <i>close</i>
          <span>取消</span>
        </button>
      </div>
      {(request.options || []).some((o) => !o.selectable) ? (
        <p className="muted small-text">
          {request.options
            .filter((o) => !o.selectable)
            .map((o) => `${o.display_name}：${o.reason}`)
            .join("；")}
        </p>
      ) : null}
    </div>
  );
}

export function ContextResult({ data, onChoose, onCancel }) {
  const r = data?.interact || {};
  const ctx = summarizeContext(data?.report);
  const final = data?.resolution || null;
  const state = data?.cancelled ? "cancelled" : final ? final.state : r.state;
  const understanding = final?.understanding || r.understanding;
  const mentions = contextMentions(data?.mentions);
  return (
    <div className="result">
      <div className="chip-row">
        <StatusChip value={state} small={false} />
        {data?.priors?.length ? (
          <span className="soft-chip" title={data.priors.join("\n")}>
            <i>history</i> 上下文 {data.priors.length} 条
          </span>
        ) : (
          <span className="soft-chip">无上下文</span>
        )}
      </div>

      {ctx && ctx.normalized && ctx.normalized !== ctx.raw ? (
        <p className="normalized">
          <span className="muted">规范化为</span> <strong>{final?.understanding?.raw_input || ctx.normalized}</strong>
        </p>
      ) : final?.understanding?.raw_input ? (
        <p className="normalized">
          <span className="muted">规范化为</span> <strong>{final.understanding.raw_input}</strong>
        </p>
      ) : null}

      {r.state === "needs_clarification" && r.request && !final && !data?.cancelled ? (
        <Clarification request={r.request} busy={data?.resolving} onChoose={onChoose} onCancel={onCancel} />
      ) : null}
      {data?.cancelled ? <p className="muted">已取消澄清，保留原始输入，未做任何解析。</p> : null}
      {final ? (
        <p className="muted small-text">
          已选择：{r.request?.options?.find((o) => o.option_id === final.answer?.option_id)?.display_name || final.answer?.option_id}
        </p>
      ) : null}
      {data?.addedPrior ? (
        <p className="added-prior">
          <i>add_circle</i> 已加入上下文：{data.addedPrior}
        </p>
      ) : null}

      <Section icon="link" title="指代解析" count={ctx?.resolutions.length ?? 0} defaultOpen={!!ctx?.resolutions.length}>
        <Resolutions items={ctx?.resolutions} />
        {mentions.length ? (
          <>
            <h6 className="sub-title">上下文中的实体</h6>
            <div className="chip-row">
              {mentions.map((m, i) => (
                <span key={i} className="soft-chip" title={`来自：${m.source}`}>
                  {m.name}
                  <span className="muted">· {m.role}</span>
                </span>
              ))}
            </div>
          </>
        ) : null}
      </Section>
      {understanding ? (
        <Section icon="account_tree" title="语义结构">
          <UnderstandingView report={understanding} compact />
        </Section>
      ) : null}
      {[...(r.diagnostics || []), ...(final?.diagnostics || [])].length ? (
        <Section icon="bug_report" title="诊断信息">
          <ul className="diag-list">
            {[...(r.diagnostics || []), ...(final?.diagnostics || [])].map((d, i) => (
              <li key={i}>{d}</li>
            ))}
          </ul>
        </Section>
      ) : null}
    </div>
  );
}

/* ------------------------------- 教学 ------------------------------- */

export function CandidateCard({ cand, decision, busy, onDecide, selectable, selected, onSelect, compact }) {
  const c = summarizeCandidate(cand);
  if (!c) return null;
  // Server status wins once it has left "proposed"; otherwise show our local outcome.
  const status = c.status !== "proposed" ? c.status : decision?.status || c.status;
  const open = status === "proposed" || status === "failed";
  // Messages saved by the earlier version carry {ok} instead of {tone}.
  const tone = decision?.tone || (decision?.ok === false ? "bad" : decision?.ok ? "good" : "info");
  return (
    <article className={`candidate-card ${selected ? "selected" : ""} ${compact ? "compact" : ""}`}>
      <div className="cand-head">
        {selectable ? (
          <label className="checkbox">
            <input type="checkbox" checked={!!selected} disabled={!open} onChange={(e) => onSelect(e.target.checked)} />
            <span />
          </label>
        ) : null}
        <div className="max">
          <div className="cand-utterance">{c.utterance}</div>
          <div className="cand-meta">
            {c.id != null ? <span>#{c.id}</span> : null}
            <span>{label(CANDIDATE_KIND, c.kind)}</span>
            {c.createdAt ? <span>{new Date(c.createdAt).toLocaleString("zh-CN")}</span> : null}
          </div>
        </div>
        <StatusChip value={status} />
      </div>
      {c.subject || c.object ? (
        <div className="atom-line">
          <span className="atom-part">{c.subject || "?"}</span>
          <span className="atom-pred">
            {c.negated ? "不" : ""}
            {label(PREDICATE, c.predicate)}
          </span>
          <span className="atom-part">{c.object || "?"}</span>
        </div>
      ) : null}
      {c.reason ? (
        <p className="cand-reason mono-soft" title={c.reason}>
          {translateReason(c.reason)}
        </p>
      ) : null}
      {c.observation && c.canConfirm && !compact ? (
        <details className="obs">
          <summary>将写入的观察</summary>
          <pre className="json-view">{JSON.stringify(c.observation, null, 2)}</pre>
        </details>
      ) : null}
      {decision?.message ? (
        <p className={`decision-msg ${tone}`} role={tone === "bad" ? "alert" : undefined}>
          <i>{tone === "good" ? "check_circle" : tone === "bad" ? "cancel" : "info"}</i>
          <span>{decision.message}</span>
        </p>
      ) : null}
      {open ? (
        <nav className="cand-actions">
          {!c.canConfirm ? <span className="warn-text small-text max">系统判断该候选无法安全写入，确认很可能失败。</span> : <span className="max" />}
          <button className="border" disabled={busy} onClick={() => onDecide(c.id, "reject")}>
            <i>close</i>
            <span>拒绝</span>
          </button>
          <button className="fill-primary" disabled={busy} onClick={() => onDecide(c.id, "confirm")}>
            {busy ? <progress className="circle small" /> : <i>check</i>}
            <span>确认写入</span>
          </button>
        </nav>
      ) : null}
    </article>
  );
}

export function TeachResult({ data, onDecide }) {
  const cand = data?.candidate;
  return (
    <div className="result">
      <p className="answer-text small">
        {cand?.id == null
          ? "未能生成候选。"
          : cand.can_confirm === false
            ? "已记录这条候选，但系统判断它暂时无法安全写入知识库。"
            : "已生成一条待审核候选。确认后才会写入知识库。"}
      </p>
      <CandidateCard cand={cand} decision={data?.decision} busy={data?.deciding} onDecide={onDecide} />
    </div>
  );
}
