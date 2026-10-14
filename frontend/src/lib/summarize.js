// Pure helpers that pull display-ready facts out of API payloads.
// They only reshape what the backend returned; they never infer new facts.

const isObj = (v) => v && typeof v === "object" && !Array.isArray(v);

export function summarizeAsk(data) {
  const v = isObj(data?.verdict) ? data.verdict : {};
  return {
    answer: data?.answer || "",
    verdict: v.verdict || "unknown",
    detail: v.detail || "",
    ambiguity: v.ambiguity || "",
    limited: !!v.limited,
    subject: v.subject_match?.text ? v.subject_match : null,
    object: v.object_match?.text ? v.object_match : null,
    queries: Array.isArray(v.queries) ? v.queries : [],
    evidence: Array.isArray(v.evidence) ? v.evidence : [],
    qkind: v.question?.qkind || "",
    unknownSpans: v.question?.unknown_spans || [],
    constructions: Array.isArray(data?.constructions) ? data.constructions : [],
    roundtrippable: !!data?.roundtrippable,
  };
}

// Merge the parse tree (roles, connective, atom) with the evaluation tree
// (status, grounding) into one renderable node tree.
export function buildTree(evalNode, exprNode, role = "") {
  if (!isObj(evalNode)) return null;
  const expr = isObj(exprNode) ? exprNode : {};
  const roles = Array.isArray(expr.roles) ? expr.roles : [];
  const exprKids = Array.isArray(expr.children) ? expr.children : [];
  const kids = Array.isArray(evalNode.children) ? evalNode.children : [];
  const g = isObj(evalNode.grounding) ? evalNode.grounding : {};
  return {
    kind: evalNode.kind || expr.kind || "",
    text: evalNode.raw_text || expr.raw_text || "",
    status: evalNode.status || "",
    role,
    connective: expr.connective || "",
    atom: isObj(expr.atom) && Object.keys(expr.atom).length ? expr.atom : null,
    verdict: g.verdict || "",
    detail: g.detail || "",
    evidence: Array.isArray(g.evidence) ? g.evidence : [],
    subject: g.subject_match?.text ? g.subject_match : null,
    object: g.object_match?.text ? g.object_match : null,
    diagnostics: Array.isArray(evalNode.diagnostics) ? evalNode.diagnostics : [],
    children: kids.map((k, i) => buildTree(k, exprKids[i], roles[i] || "")).filter(Boolean),
  };
}

export function summarizeUnderstanding(rep) {
  if (!isObj(rep)) return null;
  const counts = isObj(rep.counts) ? rep.counts : {};
  return {
    raw: rep.raw_input || "",
    complete: !!rep.complete,
    groundedAtoms: rep.grounded_atoms ?? 0,
    counts: Object.entries(counts).filter(([, n]) => n > 0),
    diagnostics: [...(rep.parse?.diagnostics || []), ...(rep.diagnostics || [])],
    atomCount: rep.parse?.atom_count ?? 0,
    tree: buildTree(rep.root, rep.parse?.tree),
  };
}

export function summarizeContext(report) {
  if (!isObj(report)) return null;
  return {
    raw: report.raw_input || "",
    normalized: report.normalized_text || "",
    complete: !!report.complete,
    resolutions: (report.resolutions || []).map((r) => ({
      mention: r.mention,
      status: r.status,
      target: r.resolved_ref?.name || "",
      targetId: r.resolved_ref?.ref_id || "",
      candidates: (r.candidates || []).map((c) => c.name || c.ref_id),
      reason: r.reason || "",
    })),
    diagnostics: report.diagnostics || [],
  };
}

export function contextMentions(context) {
  return (context?.mentions || []).map((m) => ({
    text: m.mention_text,
    name: m.object_ref?.name || m.mention_text,
    role: m.role,
    source: m.source_statement,
  }));
}

/** The statement worth remembering as a prior after an interaction turn:
 *  the normalized text (pronouns substituted) when understanding succeeded. */
export function priorFromInteraction(result, fallbackText) {
  if (!isObj(result)) return null;
  if (result.state !== "understood" && result.state !== "clarification_resolved") return null;
  const t = result.understanding?.raw_input || fallbackText || "";
  return t.trim() ? t.trim() : null;
}

export function summarizeCandidate(c) {
  if (!isObj(c)) return null;
  const p = isObj(c.parse) ? c.parse : {};
  return {
    id: c.id ?? null,
    status: c.status || "",
    kind: c.kind || "",
    utterance: c.utterance || "",
    reason: c.reason || "",
    // /api/pending rows carry no can_confirm; their kind says the same thing.
    canConfirm: c.can_confirm !== false && c.kind !== "unusable",
    createdAt: c.created_at || "",
    subject: p.subject || "",
    predicate: p.predicate || "",
    object: p.object || "",
    negated: !!p.negated,
    observation: isObj(c.observation) ? c.observation : null,
  };
}

// Backend reasons are English; translate the ones we know, keep the rest raw.
const REASONS = {
  "this shape is not safely storable yet; nothing saved": "该句式暂不支持安全写入，未保存任何内容",
  "candidate not confirmable": "该候选无法确认，未保存任何内容",
  "stored candidate unreadable": "候选数据无法读取，未保存任何内容",
  "candidate rejected, nothing written": "已拒绝，未写入任何内容",
  "already known": "知识库中已有该知识，未重复写入",
};

export function translateReason(reason) {
  const r = String(reason || "");
  if (REASONS[r]) return REASONS[r];
  const m = r.match(/^candidate is (\w+), not proposed$/);
  if (m) return `该候选已处理（${m[1] === "accepted" ? "已确认" : m[1] === "rejected" ? "已拒绝" : m[1]}），不能重复操作`;
  const n = r.match(/^no such candidate: (\d+)$/);
  if (n) return `候选 #${n[1]} 不存在`;
  return r;
}

/** Outcome of a successful confirm/reject call → {tone, status, message}. */
export function describeDecision(how, res) {
  if (how === "reject") return { tone: "info", status: "rejected", message: translateReason(res?.reason) || "已拒绝" };
  const n = res?.wrote?.length ?? 0;
  if (n === 0) return { tone: "info", status: "accepted", message: translateReason(res?.reason) || "已确认，但没有写入新内容" };
  return { tone: "good", status: "accepted", message: `已确认并写入（${n} 项）` };
}

/** A failed confirm/reject call → red state; nothing was written. */
export function describeFailure(how, err) {
  const msg = translateReason(err?.payload?.reason || err?.message) || "请求失败";
  return { tone: "bad", status: err?.offline ? "" : "failed", message: `${how === "reject" ? "拒绝" : "录入"}失败：${msg}` };
}

export function countByStatus(cands) {
  const out = { all: 0, proposed: 0, accepted: 0, rejected: 0 };
  for (const c of cands || []) {
    out.all += 1;
    if (c.status in out) out[c.status] += 1;
  }
  return out;
}

export function plainTextOf(mode, data) {
  if (!data) return "";
  if (mode === "ask") return data.answer || "";
  if (mode === "teach") return data.candidate?.reason || "";
  if (mode === "analyze") return data.report?.raw_input || "";
  if (mode === "context") return data.context?.normalized_text || data.interact?.raw_input || "";
  return "";
}
