"use strict";
const SID = "sess-" + Math.random().toString(36).slice(2, 10);
const $ = id => document.getElementById(id);
async function api(path, body) {
  const r = await fetch(path, {method: body === undefined ? "GET" : "POST",
    headers: {"Content-Type": "application/json"},
    body: body === undefined ? undefined : JSON.stringify(body)});
  return r.json();
}
function show(id, obj) {
  $(id).textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}
function appendLog(id, title, obj) {
  $(id).textContent += "\n==== " + title + " ====\n" +
    (typeof obj === "string" ? obj : JSON.stringify(obj, null, 2)) + "\n";
  $(id).scrollTop = 1e9;
}
document.querySelectorAll("#side button").forEach(b => b.onclick = () => {
  document.querySelectorAll("#side button").forEach(x => x.classList.remove("on"));
  document.querySelectorAll(".page").forEach(x => x.classList.remove("on"));
  b.classList.add("on");
  document.getElementById("p-" + b.dataset.p).classList.add("on");
});
async function refreshHeader() {
  try {
    const s = (await api("/api/status")).data;
    $("hdr-model").textContent = "模型: " + (s.active_model || "无");
    $("hdr-kb").textContent = "知识库: v" + s.kb_version + " / 概念" + s.concepts;
    $("hdr-task").textContent = "最近任务: " + (s.last_task ? s.last_task.task_id + "/" + s.last_task.status : "无");
    $("hdr-net").textContent = "服务: " + s.service;
  } catch (e) { $("hdr-net").textContent = "服务: 异常"; }
}
refreshHeader(); setInterval(refreshHeader, 10000);
$("chat-send").onclick = async () => {
  const t = $("chat-in").value; if (!t) return;
  appendLog("chat-log", "你", t);
  appendLog("chat-log", "系统", (await api("/api/chat", {text: t, session_id: SID})).data.lines.join("\n"));
  refreshHeader();
};
$("solve-send").onclick = async () => {
  const t = $("chat-in").value; if (!t) return;
  appendLog("chat-log", "解题", (await api("/api/solve", {text: t, session_id: SID})).data);
  refreshHeader();
};
let teachDraft = null;
$("t-preview").onclick = async () => {
  const attrs = {};
  ($("t-attrs").value || "").split(",").forEach(kv => {
    const i = kv.indexOf("="); if (i > 0) attrs[kv.slice(0, i).trim()] = kv.slice(i + 1).trim(); });
  const parents = ($("t-parents").value || "").split(",").map(s => s.trim()).filter(Boolean);
  const r = await api("/api/teach/preview", {name: $("t-name").value, attributes: attrs, parents});
  teachDraft = {name: $("t-name").value, attributes: attrs, parents};
  appendLog("teach-out", "预览", r);
};
$("t-confirm").onclick = async () => {
  if (!teachDraft) { appendLog("teach-out", "提示", "请先预览"); return; }
  appendLog("teach-out", "确认写入", await api("/api/teach/confirm", teachDraft));
  teachDraft = null; refreshHeader();
};
$("c-send").onclick = async () => {
  appendLog("teach-out", "纠正", await api("/api/correct",
    {name: $("c-name").value, key: $("c-key").value, value: $("c-val").value, reason: $("c-why").value || "web"}));
  refreshHeader();
};
let ruleDraft = null;
$("r-induce").onclick = async () => {
  const J = t => t.split("\n").map(s => s.trim()).filter(Boolean).map(s => JSON.parse(s));
  const r = await api("/api/rules/induce", {name: $("r-name").value || "webrule",
    domain: $("r-domain").value || "math", examples: J($("r-ex").value), negatives: J($("r-neg").value)});
  ruleDraft = r.data.saved || null;
  appendLog("rules-out", "归纳", r);
  if (r.data.ambiguous) appendLog("rules-out", "注意", "存在歧义：请先在下方选择候选（需规则id+编号），再验证。");
};
$("r-validate").onclick = async () => {
  const J = t => t.split("\n").map(s => s.trim()).filter(Boolean).map(s => JSON.parse(s));
  if (!ruleDraft) { appendLog("rules-out", "提示", "请先归纳"); return; }
  appendLog("rules-out", "验证", await api("/api/rules/validate",
    {rule_id: ruleDraft, train: J($("r-ex").value), heldout: J($("r-held").value),
     negatives: J($("r-neg").value), activate: $("r-activate").checked}));
  refreshHeader();
};
$("r-list").onclick = async () => { show("rules-list", await api("/api/rules")); };
$("q-go").onclick = async () => {
  show("concepts-out", await api("/api/concepts?q=" + encodeURIComponent($("q-q").value) + "&kind=" + encodeURIComponent($("q-kind").value)));
};
$("qq-go").onclick = async () => { show("queue-out", await api("/api/queue")); };
$("qr-ok").onclick = async () => {
  show("queue-out", await api("/api/queue/review",
    {sample_id: $("qr-id").value, accept: true, l1: $("qr-l1").value || null, l2: $("qr-l2").value || null})); };
$("qr-no").onclick = async () => {
  show("queue-out", await api("/api/queue/review", {sample_id: $("qr-id").value, accept: false})); };
$("e-go").onclick = async () => {
  show("explore-out", await api("/api/explore", {steps: parseInt($("e-steps").value || "3", 10)}));
  refreshHeader(); };
$("e-gaps").onclick = async () => { show("explore-out", await api("/api/gaps")); };
$("h-go").onclick = async () => { show("history-out", await api("/api/history?limit=30")); };
$("h-tasks").onclick = async () => { show("history-out", await api("/api/tasks?limit=20")); };
$("h-task").onclick = async () => { show("history-out", await api("/api/tasks/" + $("h-tid").value)); };
$("a-go").onclick = async () => { show("analogy-out", await api("/api/analogies")); };
$("a-fix").onclick = async () => {
  show("analogy-out", await api("/api/analogies/correct",
    {task_id: $("a-task").value, candidate_task_id: $("a-cand").value, effect: $("a-eff").value})); };
$("m-go").onclick = async () => { show("model-out", await api("/api/model")); };
$("p-go").onclick = async () => { show("model-out", await api("/api/predict", {text: $("p-text").value})); };
