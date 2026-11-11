// Single-page frontend. No build step: plain ES2020 talking to the FastAPI backend.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
let SID = null, S = null;
const COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#d03b3b"];

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
const post = (p, data) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}) });

function conf(c) { return `<span class="conf ${c < 0.85 ? "lo" : "hi"}">${Number(c).toFixed(2)}</span>`; }
function table(cols, rows) {
  return `<table><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
function nav() {
  const secs = [["sec-input", "Inputs"], ["sec-perception", "Perception"], ["sec-plan", "Plan"], ["sec-react", "ReAct + RAG"], ["sec-opt", "Bandit + QAOA"], ["sec-decision", "Decision + HITL"], ["sec-learn", "Feedback"], ["sec-audit", "Audit"]];
  const done = { "sec-input": !!SID, "sec-perception": !!S?.state, "sec-react": !!S?.trace, "sec-opt": !!S?.bandit, "sec-decision": !!S?.decision, "sec-learn": S?.decision?.status === "approved" };
  $("layernav").innerHTML = secs.map(([id, t]) => `<a href="#${id}" class="${done[id] ? "done" : ""}">${t}</a>`).join("");
}

async function health() {
  try {
    const h = await api("/health");
    $("health").innerHTML = `Qiskit ${h.qiskit} · Aer ${h.qiskit_aer} · planner: ${h.planner}<br>${esc(h.note)}`;
  } catch (e) { $("health").textContent = "backend not reachable"; }
}

async function ensureSession() { if (!SID) await newSession(); }
async function newSession() {
  S = await post("/v1/sessions"); SID = S.session_id; $("sid").textContent = SID;
  ["items", "draft", "trace", "costs", "bandit", "qubo", "hist", "qinfo", "solvers", "decision", "gantt", "review", "gt", "outcomes", "learn"].forEach((k) => ($(k).innerHTML = ""));
  $("thumb").innerHTML = "<span>No image yet</span>"; $("transcript").textContent = ""; nav();
}

async function upload(kind, file) {
  await ensureSession();
  const fd = new FormData(); fd.append("kind", kind); fd.append("file", file);
  S = await api(`/v1/sessions/${SID}/inputs`, { method: "POST", body: fd }); showInputs();
}
function showInputs() {
  if (S.inputs.image_path) $("thumb").innerHTML = `<img src="/v1/sessions/${SID}/image?t=${Date.now()}">`;
  if (S.inputs.brief_text) $("brief").value = S.inputs.brief_text;
  if (S.inputs.transcript_text) $("transcript").textContent = S.inputs.transcript_text;
  else if (S.inputs.audio_path) $("transcript").textContent = "audio attached: " + S.inputs.audio_path.split("/").pop();
}

async function perceive() {
  await ensureSession();
  try { S = await post(`/v1/sessions/${SID}/perceive`); } catch (e) { alert(e.message); return; }
  showInputs(); renderPerception(); nav();
}
function renderPerception() {
  const rows = S.perception.map((i) => [esc(i.kind), esc(i.src), `<code>${esc(JSON.stringify(i.value).slice(0, 110))}</code>`, conf(i.conf)]);
  $("items").innerHTML = table(["Kind", "Source", "Value", "Conf."], rows) + `<p class="small">Timing: ${esc(JSON.stringify(S.perception_timing))}${S.inputs.stt_method ? " · STT: " + esc(S.inputs.stt_method) : ""}</p>`;
  $("draft").textContent = JSON.stringify(S.state_draft, null, 1);
  drawOCR();
}
function drawOCR() {
  const cv = $("ocr"), ctx = cv.getContext("2d");
  if (!S.inputs.image_path) return;
  const img = new Image();
  img.onload = () => {
    const sc = cv.width / img.width; cv.height = img.height * sc;
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
    ctx.lineWidth = 2; ctx.font = "12px sans-serif";
    S.perception.filter((i) => i.kind === "task" && i.value.box).forEach((i) => {
      const [x0, y0, x1, y1] = i.value.box.map((v) => v * sc);
      ctx.strokeStyle = i.value.column === "Done" ? "#888" : "#d03b3b"; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.fillStyle = ctx.strokeStyle; ctx.fillText(`${i.src} · ${i.value.column} · ${i.conf.toFixed(2)}`, x0, Math.max(12, y0 - 4));
    });
  };
  img.src = `/v1/sessions/${SID}/image?t=${Date.now()}`;
}

async function plan() {
  await ensureSession();
  if (!S.state) await perceive();
  $("spin").classList.remove("hidden"); $("btn-plan").disabled = true;
  try {
    S = await post(`/v1/sessions/${SID}/plan`, { mode: $("mode").value, restarts: +$("restarts").value, noisy: $("noisy").checked,
      force_arm: $("arm").value || null, optimiser_sim: $("osim").value });
  } catch (e) { alert(e.message); }
  $("spin").classList.add("hidden"); $("btn-plan").disabled = false;
  renderAll();
}

function renderAll() {
  nav();
  if (S.error) $("decision").innerHTML = `<div class="flag">${esc(S.error)}</div>`;
  if (S.trace) {
    $("trace").innerHTML = `<p class="small">Planner: <b>${esc(S.planner)}</b> · status <b>${esc(S.react_status)}</b></p>` +
      table(["Hop", "Evidence needed", "Tool", "Effect on state"], S.trace.map((h) => [h.hop, esc(h.intent), `<code>${esc(h.tool)}</code>`, esc(h.effect)]));
    const st = S.state, people = st.people.map((p) => p.id);
    $("costs").innerHTML = table(["Task", "Skill", ...people.map((p) => `${p} (h)`), "Evidence (title · similarity · actual h)"], st.tasks.map((t) => [
      `${t.id} ${esc(t.title)}`, esc(t.skill), ...people.map((p) => t.est_h[p]),
      people.map((p) => `<b>${p}:</b> ` + (t.evidence[p] || []).map((e) => `${esc(e.title)} · ${e.similarity} · ${e.actual_h}`).join("; ")).join("<br>")]));
  }
  if (S.bandit) {
    const b = S.bandit;
    $("bandit").innerHTML = `<div class="kv"><div>Context</div><div><code>${esc(b.context)}</code></div><div>Mode</div><div>${esc(b.mode)}</div><div>Rule</div><div>${esc(b.rule)}</div><div>Selected arm</div><div><b>${esc(b.arm)}</b></div></div>` +
      table(["Arm", "Q", "n", "UCB score"], Object.entries(b.table).map(([a, v]) => [a === b.arm ? `<b>${a}</b>` : a, v.Q, v.n, b.scores[a] ?? "untried"]));
  }
  const q = S.qaoa;
  if (q) {
    const c = q.qubo.coefficients, is = q.ising;
    $("qubo").innerHTML = `<p class="small">Encoding <b>${q.qubo.encoding}</b> · ${q.qubo.n_qubits} qubits · μ = ${q.qubo.mu} · P = ${q.qubo.penalty_P}</p>` +
      `<p><code>C(x) = ${c.constant} ${Object.entries(c.linear).map(([k, v]) => `${v >= 0 ? "+" : "−"} ${Math.abs(v)}${k}`).join(" ")} ${Object.entries(c.quadratic).map(([k, v]) => `${v >= 0 ? "+" : "−"} ${Math.abs(v)}${k}`).join(" ")}</code></p>` +
      `<p><code>H_C = ${is.constant} I ${Object.entries(is.h).map(([k, v]) => `${v >= 0 ? "+" : "−"} ${Math.abs(v)}Z${k.slice(1)}`).join(" ")} ${Object.entries(is.J).map(([k, v]) => `${v >= 0 ? "+" : "−"} ${Math.abs(v)}Z${k.replace(/z/g, "")}`).join(" ")}</code></p>` +
      `<p class="small">Not encoded (checked in L11): ${esc(q.qubo.unencoded.join(", "))}</p>`;
    $("shots").textContent = q.shots;
    const feas = new Set(q.candidates.map((x) => x.bits)), opt = S.exact?.assignment;
    const n = q.qubo.n_qubits, keys = n <= 5 ? [...Array(2 ** n).keys()].map((k) => k.toString(2).padStart(n, "0")) : Object.keys(q.counts).sort((a, b) => q.counts[b] - q.counts[a]).slice(0, 16);
    const mx = Math.max(...Object.values(q.counts));
    const optBits = q.candidates.find((x) => JSON.stringify(x.assignment) === JSON.stringify(opt))?.bits;
    $("hist").innerHTML = keys.map((k) => { const v = q.counts[k] || 0, cls = k === optBits ? "opt" : feas.has(k) ? "feas" : "";
      return `<div class="bar"><div class="v">${v}</div><div class="b ${cls}" style="height:${(v / mx) * 85}%"></div><div class="k">${k}</div><div class="f">${cls ? (cls === "opt" ? "optimum" : "feasible") : "infeasible"}</div></div>`; }).join("");
    $("qinfo").innerHTML = `p = ${q.p} · F(γ,β) = ${q.F} · γ = [${q.gammas}] · β = [${q.betas}] · P(feasible) = ${q.P_feasible} · ${q.starts} COBYLA starts, ${q.nfev} evaluations, ${q.time_opt_s} s · circuit depth ${q.circuit.depth}, ${q.circuit.cx} CX · optimiser statevector: ${q.optimiser_sim} · <b>${esc(q.executed_on)}</b>` +
      (q.counts_noisy ? `<br>Illustrative noise model counts: <code>${esc(JSON.stringify(q.counts_noisy))}</code>` : "") +
      (q.rejected?.length ? `<br>Rejected samples: ${q.rejected.map((r) => `${r.bits} (${r.count}): ${esc(r.violations.join("; "))}`).join(" · ")}` : "");
  } else { $("qubo").innerHTML = `<p class="muted">QAOA not run for this arm.</p>`; $("hist").innerHTML = ""; $("qinfo").innerHTML = ""; }
  if (S.exact) {
    const rows = [["Brute force (exact)", esc(JSON.stringify(S.exact.assignment)), S.exact.cost, `${(S.exact.time_s * 1000).toFixed(2)} ms`, `${S.exact.n_feasible} feasible of all states`],
      ["CP-SAT (hard constraints)", esc(JSON.stringify(S.baseline.assignment)), S.baseline.cost, `${(S.baseline.time_s * 1000).toFixed(1)} ms`, S.baseline.status]];
    if (q) rows.push([`QAOA p=${q.p} (Aer)`, q.candidates[0] ? esc(JSON.stringify(q.candidates[0].assignment)) : "no feasible sample", q.candidates[0]?.cost ?? "–", `${q.time_opt_s} s optimisation`, "mode of feasible samples"]);
    $("solvers").innerHTML = table(["Solver", "Assignment", "Cost", "Time", "Note"], rows);
  }
  renderDecision(); renderLearn();
}

function renderDecision() {
  const d = S.decision; if (!d) return;
  const st = S.state, title = Object.fromEntries(st.tasks.map((t) => [t.id, t.title]));
  const box = d.status === "approved" ? `<div class="ok-box">Approved. Tickets ${esc((S.write_result?.tickets || []).join(", "))} and events created in the mock systems.</div>` :
    d.status === "rejected" ? `<div class="flag">Rejected: ${esc(d.review.comment)}</div>` : "";
  $("decision").innerHTML = box + `<div class="kv"><div>Decision</div><div><b>${d.decision_id}</b> · status <b>${d.status}</b></div>
    <div>Assignment</div><div>${Object.entries(d.assignment).map(([t, p]) => `${t} ${esc(title[t])} → <b>${p}</b>`).join("<br>")}</div>
    <div>Cost</div><div>${d.cost.total} (effort ${d.cost.effort_h} h + imbalance ${d.cost.imbalance})</div>
    <div>Load / capacity</div><div>${Object.entries(d.load_h).map(([p, h]) => `${p} ${h} / ${d.capacity_h[p]} h`).join(" · ")}</div>
    <div>Solver</div><div>${esc(d.solver.used)}${d.solver.p ? " p=" + d.solver.p : ""} · baseline ${d.solver.baseline_cost} · gap ${d.solver.gap_pct}% ${d.solver.reason ? "· " + esc(d.solver.reason) : ""}</div>
    <div>Explanation</div><div>${esc(d.explanation)}</div></div>`;
  // Gantt
  const start = 9, span = 8, pct = (hhmm) => { const [h, m] = hhmm.split(":").map(Number); return ((h + m / 60 - start) / span) * 100; };
  $("gantt").innerHTML = st.people.map((p) => `<div class="grow"><div>${p.id} (${p.avail_h} h)</div><div class="gtrack"><div class="gavail" style="width:${(p.avail_h / span) * 100}%"></div>${
    Object.entries(d.timetable).filter(([t]) => d.assignment[t] === p.id).map(([t, [a, b]], i) => `<div class="gtask" style="left:${pct(a)}%;width:${pct(b) - pct(a)}%;background:${COLORS[st.tasks.findIndex((x) => x.id === t) % 6]}">${t} ${esc(title[t])} ${a}–${b}</div>`).join("")}</div></div>`).join("") +
    `<div class="gaxis"><div></div><div class="ticks">${[...Array(span + 1).keys()].map((k) => `<span>${String(start + k).padStart(2, "0")}:00</span>`).join("")}</div></div>`;
  // Review
  if (["pending", "flagged"].includes(d.status)) {
    const flags = d.hitl.reasons || [];
    $("review").innerHTML = `<h3>${d.status === "flagged" ? "Flagged review" : "Standard review"}</h3>` +
      flags.map((f, i) => `<label class="flag"><input type="checkbox" class="ack" data-f="${esc(f)}"> ${esc(f)}</label>`).join("") +
      (d.violations?.length ? `<div class="flag">Violations: ${esc(d.violations.join("; "))}</div>` : "") +
      `<div class="row wrap">${st.tasks.map((t) => `<label class="inline">${t.id} <select class="ed" data-t="${t.id}">${st.people.map((p) => `<option ${d.assignment[t.id] === p.id ? "selected" : ""}>${p.id}</option>`).join("")}</select></label>`).join("")}
      <button class="ghost" id="btn-edit">Apply edit (re-verified)</button></div>
      <div class="row"><button class="ok" id="btn-approve">Approve and create tickets</button><input id="rej" placeholder="Reason for rejection"><button class="bad" id="btn-reject">Reject</button></div>`;
    $("btn-approve").onclick = () => doReview("approve");
    $("btn-reject").onclick = () => doReview("reject");
    $("btn-edit").onclick = () => doReview("edit");
  } else $("review").innerHTML = "";
}

async function doReview(verdict) {
  const body = { verdict, acknowledged: [...document.querySelectorAll(".ack:checked")].map((x) => x.dataset.f), comment: $("rej")?.value || "" };
  if (verdict === "edit") body.edits = Object.fromEntries([...document.querySelectorAll(".ed")].map((s) => [s.dataset.t, s.value]));
  try { S = await post(`/v1/sessions/${SID}/review`, body); } catch (e) { alert(e.message); return; }
  renderAll(); refreshAudit();
}

function renderLearn() {
  const g = S.ground_truth; if (!g) return;
  $("gt").innerHTML = `<div class="kv"><div>Level 1 exact optimum</div><div>${esc(JSON.stringify(g.level1_exact.assignment))} · cost ${g.level1_exact.cost}</div>
    <div>Proposed</div><div>${esc(JSON.stringify(g.proposed.assignment))} · gap ${(g.gap * 100).toFixed(2)} %</div>
    <div>Level 2 PM label</div><div>${esc(g.level2_label || "pending")}</div>
    <div>Level 3 outcomes</div><div>${g.level3_outcomes.length ? g.level3_outcomes.map((o) => `${o.task_id}: est ${o.estimated_h} h, actual ${o.actual_h} h`).join("<br>") + `<br>MAE ${g.effort_mae_h} h` : "not recorded"}</div></div>`;
  const d = S.decision;
  if (d.status === "approved" && !S.outcomes) {
    $("outcomes").innerHTML = Object.keys(d.assignment).map((t) => `<label class="inline">${t} actual h <input type="number" step="0.1" class="act" data-t="${t}" value="${S.state.tasks.find((x) => x.id === t).est_h[d.assignment[t]]}"></label>`).join("<br>") +
      `<div class="row"><button class="primary" id="btn-out">Record outcomes</button></div>`;
    $("btn-out").onclick = async () => {
      const outcomes = [...document.querySelectorAll(".act")].map((x) => ({ task_id: x.dataset.t, actual_h: +x.value }));
      S = await post(`/v1/sessions/${SID}/outcomes`, { outcomes }); renderAll();
    };
  } else if (S.outcomes) $("outcomes").innerHTML = `<div class="ok-box">Actual hours added to the effort history; future RAG estimates use them.</div>`;
  else $("outcomes").innerHTML = `<p class="muted">Available after approval.</p>`;
  const u = S.bandit_update, ft = S.finetune_export;
  $("learn").innerHTML = (u ? `<h3>Bandit update</h3><p>Arm <b>${u.arm}</b> in <code>${esc(u.context)}</code>: reward ${u.reward} (latency ${u.latency_s} s) · Q ${u.Q_before} → <b>${u.Q_after}</b> · n = ${u.n}</p>` : "") +
    (ft ? `<h3>Fine-tuning datasets</h3><p>SFT records added: ${ft.sft} · DPO pairs added: ${ft.dpo}${ft.reason ? " · " + esc(ft.reason) : ""} (training itself is not run in this prototype)</p>` : "");
}

async function refreshAudit() {
  const a = await api("/v1/audit?limit=25");
  $("audit").innerHTML = "<h3>Audit (latest 25)</h3>" + table(["Time", "Actor", "Action", "Ref"], a.reverse().map((e) => [e.ts.slice(11, 19), esc(e.actor), esc(e.action), esc(e.ref)]));
  const j = await api("/v1/mock/jira");
  $("jira").innerHTML = "<h3>Mock Jira tickets</h3>" + (j.length ? table(["ID", "Task", "Assignee", "Decision"], j.slice(-12).reverse().map((t) => [t.id, esc(t.title), t.assignee, t.decision_id])) : `<p class="muted">None yet.</p>`);
}

$("btn-new").onclick = newSession;
$("btn-demo").onclick = async () => { await ensureSession(); S = await post(`/v1/sessions/${SID}/demo`); showInputs(); await perceive(); };
$("btn-perceive").onclick = perceive;
$("btn-plan").onclick = plan;
$("btn-brief").onclick = async () => { await ensureSession(); const fd = new FormData(); fd.append("kind", "text"); fd.append("text", $("brief").value); S = await api(`/v1/sessions/${SID}/inputs`, { method: "POST", body: fd }); };
$("img").onchange = (e) => e.target.files[0] && upload("image", e.target.files[0]);
$("aud").onchange = (e) => e.target.files[0] && upload("audio", e.target.files[0]);
$("btn-audit").onclick = refreshAudit;
$("btn-search").onclick = async () => { const r = await post("/v1/rag/search", { query: $("q").value, k: 3 });
  $("search").innerHTML = table(["Section", "RRF", "BM25", "Dense", "Text"], r.passages.map((p) => [esc(p.section), p.rrf, p.bm25, p.dense, esc(p.text.slice(0, 160))])); };
health(); nav(); refreshAudit();
