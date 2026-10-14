import { MODES } from "../lib/labels.js";

export const SUGGESTIONS = [
  { mode: "ask", text: "杯子是不是容器？", desc: "沿继承链查证类属关系" },
  { mode: "ask", text: "杯子的材质是什么", desc: "查询属性及其验证状态" },
  { mode: "analyze", text: "杯子是容器，而且杯子可以盛水", desc: "拆分并列结构，逐项查证" },
  { mode: "context", text: "它可以盛水", priors: ["杯子是容器", "碗是容器"], desc: "指代有歧义时请你澄清" },
  { mode: "analyze", text: "如果杯子是容器，那么杯子可以盛水", desc: "识别条件句的前件与后件" },
  { mode: "teach", text: "杯子有把手", desc: "提交候选，确认后才写入" },
];

export default function Welcome({ greeting, status, onPick }) {
  return (
    <div className="welcome">
      <h1 className="greeting">
        <span className="grad" lang="en">
          {greeting}
        </span>
      </h1>
      <p className="welcome-sub">
        今天想从知识库里查些什么？
        {status?.online ? (
          <span className="muted">
            {" "}
            当前收录 {status.concepts} 个概念、{status.rules} 条规则。
          </span>
        ) : null}
      </p>
      <div className="suggestions">
        {SUGGESTIONS.map((s, i) => {
          const m = MODES[s.mode];
          return (
            <button key={i} className="suggestion" onClick={() => onPick(s)}>
              <span className="sg-text">
                {s.priors ? <span className="sg-priors">{s.priors.join(" / ")} →</span> : null}
                {s.text}
              </span>
              <span className="sg-desc">{s.desc}</span>
              <span className="sg-mode">
                <i>{m.icon}</i>
                {m.label}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
