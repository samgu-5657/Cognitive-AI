// Display vocabulary for backend enums. Unknown values fall back to the raw
// string so nothing the backend says is hidden or invented.

export const MODES = {
  ask: {
    id: "ask",
    label: "知识问答",
    short: "问答",
    icon: "help",
    placeholder: "向知识库提问，例如：杯子是不是容器？",
    hint: "只读：在知识库中查找证据并生成回答",
  },
  analyze: {
    id: "analyze",
    label: "语义分析",
    short: "分析",
    icon: "account_tree",
    placeholder: "输入一段陈述，查看它的组合结构与证据状态",
    hint: "只读：拆分并列 / 条件结构，逐个原子查证",
  },
  context: {
    id: "context",
    label: "上下文理解",
    short: "上下文",
    icon: "forum",
    placeholder: "输入一句话，系统会结合上下文解析“它”等指代",
    hint: "只读：上下文只保存在本对话中，不写入知识库",
  },
  teach: {
    id: "teach",
    label: "教学",
    short: "教学",
    icon: "school",
    placeholder: "教系统一条知识，例如：杯子有把手",
    hint: "生成待审核候选；只有确认后才会写入知识库",
  },
};

export const MODE_ORDER = ["ask", "analyze", "context", "teach"];

// tone → CSS class in styles.css (tone-*)
const STATUS = {
  supported: ["已支持", "good", "check_circle"],
  conflicted: ["有冲突", "bad", "error"],
  unknown: ["未知", "neutral", "help"],
  ambiguous: ["有歧义", "warn", "alt_route"],
  partially_supported: ["部分支持", "info", "rule"],
  unsupported: ["不支持", "neutral", "block"],
  failure: ["失败", "bad", "cancel"],
  conditional_unasserted: ["条件未断言", "info", "call_split"],
  contested: ["有争议", "warn", "report"],
  // interaction states
  understood: ["已理解", "good", "check_circle"],
  needs_clarification: ["需要澄清", "warn", "contact_support"],
  clarification_resolved: ["已澄清", "good", "task_alt"],
  invalid: ["无效", "bad", "cancel"],
  cancelled: ["已取消", "neutral", "close"],
  // reference resolution
  resolved: ["已解析", "good", "link"],
  unresolved: ["未解析", "warn", "link_off"],
  // candidates
  proposed: ["待审核", "info", "pending"],
  accepted: ["已确认", "good", "check_circle"],
  rejected: ["已拒绝", "neutral", "do_not_disturb_on"],
  failed: ["录入失败", "bad", "cancel"],
  // evidence statuses
  verified: ["已验证", "good", "verified"],
  hypothesis: ["假设", "info", "lightbulb"],
  candidate: ["候选", "neutral", "pending"],
  superseded: ["已取代", "neutral", "history"],
};

export function statusInfo(value) {
  const key = String(value ?? "");
  const hit = STATUS[key] || STATUS[key.replace(/^inherited\((.*)\)$/, "$1")];
  if (!hit) return { label: key || "—", tone: "neutral", icon: "radio_button_unchecked", raw: key };
  const inherited = /^inherited\(/.test(key);
  return { label: inherited ? `继承·${hit[0]}` : hit[0], tone: hit[1], icon: hit[2], raw: key };
}

export const EXPR_KIND = {
  atom: "原子陈述",
  conjunction: "并列",
  conditional: "条件",
  unsupported: "不支持的结构",
};

export const ROLE = {
  antecedent: "前件",
  consequent: "后件",
};

export const PREDICATE = {
  is_a: "是（类属）",
  has: "有（属性）",
  capable: "可以（能力）",
};

export const QKIND = {
  is_a_verify: "类属判断",
  capability: "能力判断",
};

export const CANDIDATE_KIND = {
  attribute: "属性",
  is_a: "类属",
  relation: "关系",
  concept: "概念",
  denial: "否定陈述",
  category_claim: "类属陈述",
  already_known: "已有知识",
  unusable: "无法写入",
};

export const label = (map, key) => map[key] || key || "—";
