import { useCallback, useEffect, useMemo, useState } from "react";
import { CandidateCard } from "./results.jsx";
import { countByStatus, describeDecision, describeFailure } from "../lib/summarize.js";

const FILTERS = [
  ["proposed", "待审核"],
  ["accepted", "已确认"],
  ["rejected", "已拒绝"],
  ["all", "全部"],
];

export default function ReviewPage({ api, notify, onChanged, onDecided }) {
  const [cands, setCands] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("proposed");
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState({});
  const [selected, setSelected] = useState(() => new Set());
  const [draft, setDraft] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [bulk, setBulk] = useState(false);
  const [outcomes, setOutcomes] = useState({});

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.pending();
      setCands([...(data.candidates || [])].sort((a, b) => (b.id ?? 0) - (a.id ?? 0)));
      setError("");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    load();
  }, [load]);

  const counts = useMemo(() => countByStatus(cands), [cands]);
  const q = query.trim();
  const shown = cands.filter((c) => (filter === "all" || c.status === filter) && (!q || (c.utterance || "").includes(q)));
  const openShown = shown.filter((c) => c.status === "proposed");

  const decideOne = async (id, how) => {
    setBusy((b) => ({ ...b, [id]: true }));
    try {
      const res = await api[how](id);
      onDecided(id, how, res);
      setOutcomes((o) => ({ ...o, [id]: describeDecision(how, res) }));
      return true;
    } catch (e) {
      const d = describeFailure(how, e);
      setOutcomes((o) => ({ ...o, [id]: d }));
      notify({ text: `#${id} ${d.message}`, tone: "error" });
      return false;
    } finally {
      setBusy((b) => ({ ...b, [id]: false }));
    }
  };

  const decide = async (id, how) => {
    const ok = await decideOne(id, how);
    if (ok) notify({ text: how === "confirm" ? `#${id} 已确认` : `#${id} 已拒绝`, icon: how === "confirm" ? "check_circle" : "info" });
    setSelected((s) => {
      const n = new Set(s);
      n.delete(id);
      return n;
    });
    await load();
    onChanged();
  };

  const decideBulk = async (how) => {
    const ids = [...selected];
    if (!ids.length) return;
    setBulk(true);
    let ok = 0;
    for (const id of ids) if (await decideOne(id, how)) ok += 1;
    setBulk(false);
    setSelected(new Set());
    const failed = ids.length - ok;
    notify(
      failed
        ? { text: `${ok} 条成功，${failed} 条失败（失败原因见各卡片）`, tone: "error" }
        : { text: `已${how === "confirm" ? "确认" : "拒绝"} ${ok} 条候选`, icon: "check_circle" },
    );
    await load();
    onChanged();
  };

  const submit = async () => {
    const t = draft.trim();
    if (!t) return;
    setSubmitting(true);
    try {
      const res = await api.propose(t);
      setDraft("");
      notify({ text: `已提交候选 #${res.candidate?.id ?? "?"}`, icon: "school" });
      setFilter("proposed");
      await load();
      onChanged();
    } catch (e) {
      notify({ text: e.message, tone: "error" });
    } finally {
      setSubmitting(false);
    }
  };

  const allSelected = openShown.length > 0 && openShown.every((c) => selected.has(c.id));

  return (
    <div className="review-page">
      <div className="review-head">
        <div className="max">
          <h4>知识审核</h4>
          <p className="muted">教学模式生成的候选在这里统一审核。确认后写入知识库，拒绝则不写入任何内容。</p>
        </div>
        <button className="border" onClick={load} disabled={loading}>
          <i>refresh</i>
          <span>刷新</span>
        </button>
      </div>

      <div className="teach-box">
        <i>school</i>
        <input
          value={draft}
          placeholder="提交一条新知识候选，例如：碗有把手"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.nativeEvent.isComposing && submit()}
        />
        <button className="fill-primary" disabled={!draft.trim() || submitting} onClick={submit}>
          {submitting ? <progress className="circle small" /> : <i>add</i>}
          <span>提交候选</span>
        </button>
      </div>

      <div className="review-toolbar">
        <nav className="seg-buttons" role="tablist">
          {FILTERS.map(([k, name]) => (
            <button
              key={k}
              role="tab"
              aria-selected={filter === k}
              className={filter === k ? "on" : ""}
              onClick={() => {
                setFilter(k);
                setSelected(new Set());
              }}
            >
              {filter === k ? <i>check</i> : null}
              <span>{name}</span>
              <span className="seg-count">{counts[k]}</span>
            </button>
          ))}
        </nav>
        <div className="side-search review-search">
          <i>search</i>
          <input value={query} placeholder="搜索候选" onChange={(e) => setQuery(e.target.value)} />
        </div>
      </div>

      {openShown.length ? (
        <div className="bulk-bar">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={allSelected}
              onChange={(e) => setSelected(e.target.checked ? new Set(openShown.map((c) => c.id)) : new Set())}
            />
            <span>{selected.size ? `已选 ${selected.size} 条` : "全选待审核"}</span>
          </label>
          <span className="max" />
          {selected.size ? (
            <>
              <button className="border small" disabled={bulk} onClick={() => decideBulk("reject")}>
                <i>close</i>
                <span>批量拒绝</span>
              </button>
              <button className="fill-primary small" disabled={bulk} onClick={() => decideBulk("confirm")}>
                <i>done_all</i>
                <span>批量确认</span>
              </button>
            </>
          ) : null}
        </div>
      ) : null}

      {error ? (
        <div className="error-box">
          <i>cloud_off</i>
          <span className="max">{error}</span>
          <button className="border small" onClick={load}>
            重试
          </button>
        </div>
      ) : null}
      {loading && !cands.length ? <progress className="max" /> : null}

      <div className="cand-list">
        {shown.map((c) => (
          <CandidateCard
            key={c.id}
            cand={c}
            decision={outcomes[c.id]}
            busy={!!busy[c.id] || bulk}
            onDecide={decide}
            selectable
            selected={selected.has(c.id)}
            onSelect={(on) =>
              setSelected((s) => {
                const n = new Set(s);
                on ? n.add(c.id) : n.delete(c.id);
                return n;
              })
            }
          />
        ))}
        {!loading && !error && !shown.length ? (
          <div className="empty-state">
            <i>inventory_2</i>
            <p>{filter === "proposed" ? "没有待审核的候选" : "这里还没有内容"}</p>
          </div>
        ) : null}
      </div>
    </div>
  );
}
