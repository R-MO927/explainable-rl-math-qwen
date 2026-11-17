/* MathMates front end. Everything shown comes from /api/solve (the real pipeline) or /api/results
   (the saved evaluation files). Nothing here invents numbers. */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const t = (k, v) => I18N.t(k, v);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const EXAMPLES = [
    "Mia picks 4 baskets of apples with 6 apples in each basket. She eats 3 apples. How many apples are left?",
    "A train travels 60 km per hour for 2.5 hours. How many kilometers does it travel?",
    "Lina had 3 dozen eggs. She used 14 eggs for baking and broke 2. How many eggs are left?",
    "Omar has 18 balloons. He gives 1/3 of them to his sister and then buys 7 more. How many balloons does he have now?",
  ];
  const ITEMS = ["🍎", "🍊", "🍓", "🍌", "🍇", "🍉", "🍒", "🍐", "🧸", "⚽", "🎈", "🍪", "⭐", "🧁"];
  const METHOD_KEYS = ["A_single_attempt", "B_fixed_rule", "C_rl", "ref_max_budget"];
  const METHOD_CLASS = { A_single_attempt: "A", B_fixed_rule: "B", C_rl: "C", ref_max_budget: "R" };
  const TIMEOUT_MS = 5 * 60 * 1000;

  let lastResult = null, results = null, split = "test", health = null, busy = false;

  // ---------------------------------------------------------------- math text rendering
  function tex(body, display) {
    try { return katex.renderToString(body, { displayMode: display, throwOnError: false, output: "html", strict: "ignore" }); }
    catch (e) { return `<code>${esc(body)}</code>`; }
  }
  const BOXED = /\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/g;
  function plain(s) {
    let out = "", last = 0, m;
    BOXED.lastIndex = 0;
    while ((m = BOXED.exec(s))) { out += inline(s.slice(last, m.index)); out += tex(`\\boxed{${m[1]}}`, false); last = BOXED.lastIndex; }
    return out + inline(s.slice(last));
  }
  function inline(s) {
    return esc(s)
      .replace(/^\s*#{1,6}\s*/gm, "")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\n/g, "<br>");
  }
  const MATH = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\))/g;
  function rich(text) {
    let out = "", last = 0, m;
    MATH.lastIndex = 0;
    while ((m = MATH.exec(text))) {
      out += plain(text.slice(last, m.index));
      const tok = m[0];
      const display = tok.startsWith("$$") || tok.startsWith("\\[");
      out += tex(tok.slice(2, -2), display);
      last = MATH.lastIndex;
    }
    return out + plain(text.slice(last));
  }
  function splitSteps(text) {
    let parts = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
    if (parts.length < 2) {            // single block: split by lines, keep \[ ... \] together
      parts = [];
      let buf = "";
      for (const line of text.split("\n")) {
        buf = buf ? buf + "\n" + line : line;
        const open = (buf.match(/\\\[/g) || []).length > (buf.match(/\\\]/g) || []).length;
        if (!open && buf.trim()) { parts.push(buf.trim()); buf = ""; }
      }
      if (buf.trim()) parts.push(buf.trim());
    }
    return parts;
  }

  // ---------------------------------------------------------------- health
  async function pollHealth() {
    try {
      const r = await fetch("/api/health", { cache: "no-store" });
      health = await r.json();
    } catch (e) { health = { status: "offline" }; }
    renderHealth();
    if (!health || health.status === "loading" || health.status === "offline") setTimeout(pollHealth, 2000);
  }
  function renderHealth() {
    const pill = $("#modelPill"), txt = $("#modelPillText");
    pill.className = "pill";
    if (!health || health.status === "loading") { pill.classList.add("pill-loading"); txt.textContent = t("modelLoading"); }
    else if (health.status === "ready") {
      pill.classList.add(health.demo ? "pill-demo" : "pill-ready");
      txt.textContent = health.demo ? t("modelDemo") : t("modelReady");
      pill.title = health.model ? `${health.model.model_id || ""} ${health.model.device || ""}` : "";
    } else { pill.classList.add("pill-error"); txt.textContent = health.status === "offline" ? t("err_network") : t("modelError"); pill.title = health.error || ""; }
    $("#demoBanner").hidden = !(health && health.demo);
    const sn = $("#speedNote");
    if (health && health.site === "space" && !health.demo) { sn.dataset.i18n = "speedNoteSpace"; sn.textContent = t("speedNoteSpace"); }
    else if (health && health.model && health.model.device === "cpu" && !health.demo) { sn.dataset.i18n = "speedNoteCpu"; sn.textContent = t("speedNoteCpu"); }
    $("#solveBtn").disabled = busy || !health || health.status !== "ready";
  }

  // ---------------------------------------------------------------- solve
  let timerId = null, msgId = null;
  function showLoading(on) {
    $("#loading").hidden = !on;
    clearInterval(timerId); clearInterval(msgId);
    if (on) {
      const t0 = performance.now(); let i = 0;
      $("#elapsed").textContent = "0";
      $("#loadingMsg").textContent = t("loading1");
      timerId = setInterval(() => { $("#elapsed").textContent = Math.floor((performance.now() - t0) / 1000); }, 250);
      msgId = setInterval(() => { i = (i + 1) % 4; $("#loadingMsg").textContent = t(`loading${i + 1}`); Sfx.play("blip"); }, 4000);
      $("#loading").scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }
  function showError(msg) {
    $("#errorBox").hidden = false;
    $("#errorText").textContent = msg;
  }

  async function solve() {
    const question = $("#question").value.trim();
    $("#errorBox").hidden = true;
    if (!question) { Sfx.play("error"); return showError(t("err_empty_question")); }
    Sfx.play("launch");
    const method = ($("input[name=method]:checked") || {}).value || "rl";
    const budget = parseInt($("#budget").value, 10);
    busy = true; renderHealth();
    $("#solveBtn").classList.add("is-busy");
    $("#result").hidden = true;
    showLoading(true);
    const ctrl = new AbortController();
    const kill = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
      const r = await fetch("/api/solve", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, method, budget }), signal: ctrl.signal,
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        Sfx.play("error");
        const key = `err_${data.error}`;
        showError(I18N.t(key) !== key ? t(key) : t("err_generic", { d: data.detail || data.error || r.status }));
      } else {
        lastResult = data;
        renderResult();
        if (data.final_answer != null) { burst(); Sfx.play("success"); Sfx.play("sparkle"); }   // celebrates finishing, not correctness
        else Sfx.play("uhoh");
      }
    } catch (e) {
      Sfx.play("error");
      showError(e.name === "AbortError" ? t("err_timeout") : t("err_network"));
    } finally {
      clearTimeout(kill);
      busy = false;
      $("#solveBtn").classList.remove("is-busy");
      showLoading(false);
      renderHealth();
    }
  }

  // ---------------------------------------------------------------- result rendering
  function renderResult() {
    const r = lastResult;
    if (!r) return;
    $("#result").hidden = false;
    renderAnswer(r);
    renderSteps(r);
    renderFruits(r);
    renderStory(r);
    renderTries(r);
    if (!renderResult.scrolled) { $("#result").scrollIntoView({ behavior: "smooth", block: "start" }); }
  }

  function renderAnswer(r) {
    const has = r.final_answer !== null && r.final_answer !== undefined;
    $("#answerValue").innerHTML = has ? `<bdi dir="ltr">${esc(r.final_answer)}</bdi>` :`<span class="no-ans">${esc(t("noAnswer"))}</span>`;
    const answered = r.attempt_answers.filter((a) => a !== null);
    const agree = has ? r.attempt_answers.filter((a) => a === r.final_answer).length : 0;
    const badges = [];
    if (!has) {
      const words = nonNumericBoxed(r.attempt_texts);
      badges.push(`<span class="badge bad">${esc(words ? t("noAnswerWords", { w: words }) : t("noAnswerExplain"))}</span>`);
    }
    if (has && r.attempt_answers.length > 1) badges.push(`<span class="badge">🗳️ ${esc(t("badgeVote", { k: agree, n: r.attempt_answers.length }))}</span>`);
    const pot = r.verifier_outputs && r.verifier_outputs.pot;
    if (has && pot && pot.pot_answer != null) badges.push(pot.pot_answer === r.final_answer
      ? `<span class="badge good">🦉 ${esc(t("badgePotOk"))}</span>` : `<span class="badge warn">🦉 ${esc(t("badgePotBad"))}</span>`);
    const shown = r.display.shown_attempt;
    const basic = shown != null ? r.verifier_outputs.basic[shown] : null;
    if (has && basic) {
      if (basic.arith_status === "all_consistent") badges.push(`<span class="badge good">🧮 ${esc(t("badgeArithOk"))}</span>`);
      if (basic.arith_status === "error_found") badges.push(`<span class="badge warn">🧮 ${esc(t("badgeArithBad"))}</span>`);
    }
    $("#answerBadges").innerHTML = badges.join("");
    const u = r.usage;
    const wall = r.latency_s && r.latency_s.wall_clock != null ? r.latency_s.wall_clock : r.server_time_s;
    $("#usage").innerHTML = [
      ["🔁", u.attempts, t("usageTries")], ["🤖", u.model_calls, t("usageCalls")],
      ["⏱️", Number(wall).toFixed(1), t("usageTime")], ["✍️", u.output_tokens, t("usageTokens")],
    ].map(([i, v, l]) => `<div class="u"><span class="u-ico">${i}</span><b>${esc(v)}</b><small>${esc(l)}</small></div>`).join("");
    $(".answer-card").classList.toggle("is-empty", !has);
  }

  // What Robo put in the box when it was not a number (e.g. "Paris"), for an honest message.
  function nonNumericBoxed(texts) {
    for (let i = texts.length - 1; i >= 0; i--) {
      const all = [...texts[i].matchAll(/\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}/g)];
      if (all.length) {
        const raw = all[all.length - 1][1].replace(/\\text\{([^{}]*)\}/g, "$1").replace(/\\[a-zA-Z]+/g, "").replace(/[{}]/g, "").trim();
        if (raw && !/\d/.test(raw)) return raw.slice(0, 40);
      }
    }
    return null;
  }

  function burst() {
    const box = $("#confetti");
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    box.innerHTML = "";
    const sy = ["⭐", "✨", "+", "×", "=", "🎉", "÷", "π"];
    for (let i = 0; i < 26; i++) {
      const s = document.createElement("span");
      s.textContent = sy[i % sy.length];
      s.style.setProperty("--x", `${(Math.random() - 0.5) * 520}px`);
      s.style.setProperty("--y", `${-120 - Math.random() * 200}px`);
      s.style.setProperty("--r", `${(Math.random() - 0.5) * 720}deg`);
      s.style.animationDelay = `${Math.random() * 0.15}s`;
      box.appendChild(s);
    }
    setTimeout(() => (box.innerHTML = ""), 1800);
  }

  function renderSteps(r) {
    const i = r.display.shown_attempt;
    const ol = $("#steps");
    if (i == null) { ol.innerHTML = ""; $("#stepsFrom").textContent = t("noAnswerExplain"); return; }
    $("#stepsFrom").textContent = r.final_answer == null ? t("stepsFromNoAns", { n: i + 1 }) : t("stepsFrom", { n: i + 1 });
    const parts = splitSteps(r.attempt_texts[i]);
    ol.innerHTML = parts.map((p, k) => {
      const fin = /\\boxed/.test(p);
      return `<li class="step${fin ? " final-step" : ""}" style="--d:${k * 70}ms"><span class="step-num">${fin ? "🏆" : k + 1}</span><div class="step-body">${rich(p)}</div></li>`;
    }).join("");
  }

  // ---- fruits
  function group(n, icon, goneFrom = Infinity) {
    let s = "";
    for (let k = 0; k < n; k++) s += `<span class="it${k >= goneFrom ? " gone" : ""}">${icon}</span>`;
    return `<span class="grp">${s}</span>`;
  }
  function visualFor(d, idx) {
    const icon = ITEMS[idx % ITEMS.length];
    const flat = d.lhs.replace(/\s+/g, "");
    // "a/b of n": n items with the picked part highlighted
    const fr = flat.match(/^\(\((\d+)\)\/\((\d+)\)\)\*(\d+)$/) || flat.match(/^(\d+)\*\(\((\d+)\)\/\((\d+)\)\)$/);
    if (fr) {
      const [a, b, n] = flat.startsWith("((") ? [+fr[1], +fr[2], +fr[3]] : [+fr[2], +fr[3], +fr[1]];
      const pick = (n * a) / b;
      if (b > 0 && n <= 30 && Number.isInteger(pick)) {
        let s = "";
        for (let k = 0; k < n; k++) s += `<span class="it${k < pick ? " pick" : " dim"}">${icon}</span>`;
        return `<div class="vis"><span class="grp">${s}</span></div>`;
      }
    }
    const m = flat.match(/^(\d+)([+\-*/])(\d+)$/);
    if (m) {
      const a = +m[1], op = m[2], b = +m[3];
      if (op === "+" && a > 0 && b > 0 && a + b <= 30)
        return `<div class="vis">${group(a, icon)}<span class="op">+</span>${group(b, icon)}</div>`;
      if (op === "-" && a > 0 && a <= 30 && b <= a)
        return `<div class="vis">${group(a, icon, a - b)}</div>`;
      if (op === "*" && a > 0 && b > 0 && a <= 10 && b <= 10 && a * b <= 40)
        return `<div class="vis baskets">${Array.from({ length: a }, () => `<span class="basket">${group(b, icon)}</span>`).join("")}</div>`;
      if (op === "/" && b > 0 && b <= 10 && a <= 40 && a % b === 0)
        return `<div class="vis baskets">${Array.from({ length: b }, () => `<span class="basket plate">${group(a / b, icon)}</span>`).join("")}</div>`;
    }
    if (m) {   // bigger numbers, still pictured: baskets with a count, or fruit bags
      const a = +m[1], op = m[2], b = +m[3];
      if (op === "*" && Number.isInteger(a) && a >= 1 && a <= 10)
        return `<div class="vis baskets">${Array.from({ length: a }, () => `<span class="basket bag"><span class="it">${icon}</span><b>${esc(m[3])}</b></span>`).join("")}</div>`;
      if (op === "*" && Number.isInteger(b) && b >= 1 && b <= 10)
        return `<div class="vis baskets">${Array.from({ length: b }, () => `<span class="basket bag"><span class="it">${icon}</span><b>${esc(m[1])}</b></span>`).join("")}</div>`;
    }
    // general case: cartoon number blocks, each carrying a fruit
    const toks = d.lhs.match(/\d+(?:\.\d+)?|[+\-*/()]/g) || [d.lhs];
    const sym = { "*": "×", "/": "÷", "-": "−", "+": "+" };
    return `<div class="vis blocks">${toks.map((x) => /\d/.test(x)
      ? `<span class="nblock"><i>${icon}</i>${esc(x)}</span>` : `<span class="op">${esc(sym[x] || x)}</span>`).join("")}<span class="op">=</span><span class="nblock res"><i>${icon}</i>${esc(d.lhs_value)}</span></div>`;
  }
  function texOf(d) {
    const l = d.lhs.replace(/\s+/g, "")
      .replace(/\(\((\d+(?:\.\d+)?)\)\/\((\d+(?:\.\d+)?)\)\)/g, "\\tfrac{$1}{$2}")
      .replace(/\*/g, " \\times ").replace(/\//g, " \\div ").replace(/-/g, " - ").replace(/\+/g, " + ");
    return tex(`${l} = ${d.rhs}`, false);
  }
  function renderFruits(r) {
    const i = r.display.shown_attempt;
    const list = i == null ? [] : (r.display.arithmetic[i] || []);
    const box = $("#fruits");
    if (!list.length) { box.innerHTML = `<div class="empty">${Mascots.svg("hoot")}<p>${esc(t("noFruits"))}</p></div>`; return; }
    box.innerHTML = list.slice(0, 8).map((d, k) => `
      <div class="fruit-card ${d.consistent ? "ok" : "bad"}" style="--d:${k * 90}ms">
        <div class="fruit-formula" dir="ltr">${texOf(d)}</div>
        ${visualFor(d, k)}
        <div class="fruit-check">${d.consistent ? esc(t("checkedOk")) : esc(t("checkedBad", { v: d.lhs_value }))}</div>
      </div>`).join("");
  }

  // ---- story (XAI)
  function ruleText(rule) {
    if (!rule) return "";
    const map = [[/^1:/, "rule_1"], [/^2:/, "rule_2"], [/^3:/, "rule_3"], [/^4:/, "rule_4"], [/^5:/, "rule_5"],
                 [/^single attempt/, "rule_single"], [/^use all attempts/, "rule_all"], [/^use PoT/, "rule_pot"], [/^budget exhausted/, "rule_exhausted"]];
    const hit = map.find(([re]) => re.test(rule));
    return hit ? t(hit[1]) : rule;
  }
  function situation(d) {
    const st = d.state, n = st.n_attempts;
    const tries = n === 1 ? t("triesWord1") : t("triesWordN");
    const p = t(`p_${st.pot_check}`);
    if (d.candidate == null) return t("sitNoCand", { n, tries, p });
    const s = st.support;
    return t("sit", { n, tries, c: `<b dir="ltr">${esc(d.candidate)}</b>`, s, agree: s === 1 ? t("agreeWord1") : t("agreeWordN"),
                      a: t(`a_${st.candidate_arithmetic}`), p });
  }
  function qBars(d) {
    if (!d.q_values) return "";
    const entries = Object.entries(d.q_values);
    const max = Math.max(0.05, ...entries.filter(([, v]) => v != null).map(([, v]) => Math.abs(v)));
    return `<div class="qbars" dir="ltr">${entries.map(([a, v]) => {
      if (v == null) return `<div class="qrow off"><span class="qlab">${esc(t(`actShort_${a}`))}</span><span class="qtrack"><span class="qzero"></span></span><span class="qval">${esc(t("unavailable"))}</span></div>`;
      const w = (Math.abs(v) / max) * 50;
      const side = v >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
      const chosen = a === d.chosen_action;
      return `<div class="qrow${chosen ? " chosen" : ""}${v < 0 ? " neg" : ""}"><span class="qlab">${esc(t(`actShort_${a}`))}${chosen ? ` · ${esc(t("chosen"))}` : ""}</span>
        <span class="qtrack"><span class="qzero"></span><span class="qfill" style="${side}"></span></span>
        <span class="qval">${v >= 0 ? "+" : ""}${v.toFixed(3)}</span></div>`;
    }).join("")}<p class="qnote">${esc(t("qNote"))}</p></div>`;
  }
  function renderStory(r) {
    const who = r.method === "rl" ? ["captain", t("speakerCap")] : r.method === "single" ? ["robo", t("speakerRobo")] : ["hoot", t("speakerRule")];
    $("#story").innerHTML = r.explanations.map((d, k) => {
      let why = "";
      const nAvail = (d.available_actions || []).length;
      if (d.fallback) why = `${t("whyFallback")} ${t("whyRule", { r: ruleText(d.rule) })}`;
      else if (d.q_values && nAvail > 1) why = t("whyQ", { m: (d.margin_over_next_best ?? 0).toFixed(3) });
      else if (d.q_values) why = t("whyOnly");
      else why = t("whyRule", { r: ruleText(d.rule) });
      return `<div class="bubble-row" style="--d:${k * 120}ms">
        <div class="avatar">${Mascots.svg(who[0])}</div>
        <div class="bubble">
          <div class="bubble-head"><span class="speaker">${esc(who[1])}</span><span class="move move-${esc(d.chosen_action)}">${esc(t(`actShort_${d.chosen_action}`))}</span><span class="stepno">#${k + 1}</span></div>
          <p class="say">“${esc(t(`act_${d.chosen_action}`))}”</p>
          <p class="sit">${situation(d)}</p>
          <p class="why">💡 ${esc(why)}</p>
          ${qBars(d)}
        </div></div>`;
    }).join("");
    $("#techLog").textContent = r.explanations.map((d, k) => `#${k + 1} ${d.text}`).join("\n\n") + `\n\n${r.note || ""}`;
  }

  // ---- tries + PoT
  function renderTries(r) {
    const basic = r.verifier_outputs.basic;
    $("#tries").innerHTML = r.attempt_texts.map((txt, i) => {
      const b = basic[i] || {};
      const ans = r.attempt_answers[i];
      const star = i === r.display.shown_attempt ? " shown" : "";
      return `<div class="try${star}">
        <div class="try-head"><span class="try-n">${esc(t("tryN", { n: i + 1 }))}</span>
          <span class="try-kind">${esc(i === 0 ? t("tryGreedy") : t("trySampled"))}</span>
          <span class="try-ans">${esc(t("answerIs"))}: <b dir="ltr">${esc(ans ?? t("none"))}</b></span>
          <span class="badge ${b.arith_status === "error_found" ? "warn" : b.arith_status === "all_consistent" ? "good" : ""}">${esc(t(`arith_${b.arith_status}`))}</span>
          ${b.truncated ? `<span class="badge warn">${esc(t("truncated"))}</span>` : ""}
          <button type="button" class="link toggle" data-i="${i}">${esc(t("showText"))}</button></div>
        <div class="try-text" dir="ltr" hidden>${rich(txt)}</div></div>`;
    }).join("");
    $$("#tries .toggle").forEach((btn) => btn.addEventListener("click", () => {
      const box = btn.closest(".try").querySelector(".try-text");
      box.hidden = !box.hidden;
      btn.textContent = box.hidden ? t("showText") : t("hideText");
    }));
    const pot = r.verifier_outputs.pot;
    const pb = $("#potBox");
    if (!pot) { pb.hidden = true; return; }
    pb.hidden = false;
    const code = (r.pot_program_output || "").replace(/^[\s\S]*?```(?:python)?\n?/, "").replace(/```[\s\S]*$/, "");
    const verdict = pot.pot_answer == null ? t("potFailed") : (pot.pot_answer === r.final_answer ? t("potAgree") : t("potDisagree"));
    pb.innerHTML = `<div class="pot-head">${Mascots.svg("hoot")}<h3>${esc(t("potTitle"))}</h3></div>
      <pre class="code" dir="ltr">${esc(code.trim() || r.pot_program_output || "")}</pre>
      <p>${esc(t("potResult"))}: <b dir="ltr">${esc(pot.pot_answer ?? "—")}</b> · ${esc(verdict)}</p>`;
  }

  // ---------------------------------------------------------------- results section
  async function loadResults() {
    try { results = await (await fetch("/api/results")).json(); } catch (e) { results = null; }
    renderResults();
  }
  const pct = (x) => (x * 100).toFixed(1);
  function renderResults() {
    const R = results && results[split];
    $("#resSub").textContent = split === "test" ? t("resSub") : t("resSubSmall");
    if (!R) { $("#resTiles").innerHTML = ""; return; }
    $("#resTiles").innerHTML = ["A_single_attempt", "B_fixed_rule", "C_rl"].map((k, i) => {
      const m = R.methods[k];
      return `<div class="card tile tile-${METHOD_CLASS[k]}" style="--d:${i * 100}ms">
        <div class="tile-mascots">${k === "A_single_attempt" ? Mascots.svg("robo") : k === "B_fixed_rule" ? Mascots.svg("robo") + Mascots.svg("hoot") : Mascots.svg("robo") + Mascots.svg("hoot") + Mascots.svg("captain")}</div>
        <h3><span class="sw"></span>${esc(t(k))}</h3>
        <div class="big" dir="ltr">${pct(m.accuracy)}<small>%</small></div>
        <div class="muted">${esc(t("accuracy"))} · ${esc(t("correctOf", { k: m.n_correct, n: m.n }))}</div>
        <div class="muted small" dir="ltr">${esc(t("ci"))}: ${pct(m.accuracy_ci95_wilson[0])}–${pct(m.accuracy_ci95_wilson[1])}%</div>
        <div class="tile-cost"><b dir="ltr">${m.mean_model_calls.toFixed(2)}</b> ${esc(t("callsPer"))}</div></div>`;
    }).join("");
    $("#costBars").innerHTML = METHOD_KEYS.map((k) => {
      const v = R.methods[k].mean_model_calls;
      return `<div class="cbar cbar-${METHOD_CLASS[k]}"><span class="clab">${esc(t(k))}</span>
        <span class="ctrack"><span class="cfill" style="width:${(v / 5) * 100}%"></span></span><span class="cval" dir="ltr">${v.toFixed(2)}</span></div>`;
    }).join("");
    const c = R.comparisons, A = R.methods.A_single_attempt, B = R.methods.B_fixed_rule, C = R.methods.C_rl;
    const runs = Object.values(R.seed_runs || {}).map((s) => s.accuracy);
    const pts = (x) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}`;
    const items = [
      t("take1", { d: pts(c["B_fixed_rule - A_single_attempt"].diff), x: (B.mean_model_calls / A.mean_model_calls).toFixed(1) }),
      t("take2", { d: pts(c["C_rl - A_single_attempt"].diff), x: (C.mean_model_calls / A.mean_model_calls).toFixed(1) }),
      t("take3", { d: pts(c["C_rl - B_fixed_rule"].diff) }),
    ];
    if (runs.length) items.push(t("take4", { lo: pct(Math.min(...runs)), hi: pct(Math.max(...runs)) }));
    items.push(t("take5"));
    $("#takeaways").innerHTML = items.map((s) => `<li>${esc(s)}</li>`).join("");
  }

  // ---------------------------------------------------------------- wiring
  function wire() {
    const q = $("#question");
    const onInput = () => {
      $("#charCount").textContent = `${q.value.length} / 1000`;
      $("#arabicWarn").hidden = !/[؀-ۿ]/.test(q.value);
    };
    q.addEventListener("input", onInput);
    q.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) solve(); });
    $$(".chip[data-example]").forEach((b) => b.addEventListener("click", () => { Sfx.play("pop"); q.value = EXAMPLES[+b.dataset.example]; onInput(); q.focus(); }));
    $("#budget").addEventListener("input", (e) => { $("#budgetVal").textContent = e.target.value; Sfx.play("tick"); });
    $$("input[name=method]").forEach((r) => r.addEventListener("change", () => Sfx.play("select")));
    $("#solveBtn").addEventListener("click", solve);
    $("#langBtn").addEventListener("click", () => { Sfx.play("swoosh"); I18N.set(I18N.lang === "en" ? "ar" : "en"); });
    const muteBtn = $("#muteBtn");
    const showMute = () => { muteBtn.textContent = Sfx.muted ? "🔇" : "🔊"; muteBtn.setAttribute("aria-pressed", String(Sfx.muted)); muteBtn.title = t(Sfx.muted ? "soundOff" : "soundOn"); };
    muteBtn.addEventListener("click", () => { Sfx.toggle(); showMute(); });
    document.addEventListener("langchange", showMute);
    showMute();
    $$(".tab").forEach((tab) => tab.addEventListener("click", () => {
      Sfx.play("tab");
      $$(".tab").forEach((x) => x.classList.toggle("active", x === tab));
      $$(".tab-panel").forEach((p) => (p.hidden = p.dataset.panel !== tab.dataset.tab));
      if (tab.dataset.tab === "fruits") {   // a pop for each fruit card as it appears
        const n = Math.min(6, $$("#fruits .fruit-card").length);
        for (let k = 0; k < n; k++) setTimeout(() => Sfx.play("pop"), 90 * k + 60);
      }
    }));
    // generic click sound for links/buttons that have no special sound
    document.addEventListener("click", (e) => {
      const el = e.target.closest(".nav a, .btn-primary, .btn-ghost, .link, .brand");
      if (el) Sfx.play("click");
      // tap a mascot: it jumps and says hello in its own voice
      const m = e.target.closest(".mascot-svg");
      if (m) {
        const voice = { Robo: "robo", Hoot: "hoot", "Captain Q": "captain" }[m.getAttribute("aria-label")];
        if (voice) Sfx.play(voice);
        m.classList.remove("jump"); void m.getBoundingClientRect(); m.classList.add("jump");
      }
    });
    $$(".seg[data-split]").forEach((b) => b.addEventListener("click", () => {
      Sfx.play("tab");
      split = b.dataset.split;
      $$(".seg[data-split]").forEach((x) => x.classList.toggle("active", x === b));
      renderResults();
    }));
    document.addEventListener("langchange", () => {
      renderHealth(); renderResults();
      if (lastResult) { renderResult.scrolled = true; renderResult(); renderResult.scrolled = false; }
    });
    // reveal-on-scroll
    const io = "IntersectionObserver" in window ? new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }), { threshold: 0.12 }) : null;
    $$(".section, .solve-card, .hero").forEach((el) => { el.classList.add("reveal"); io ? io.observe(el) : el.classList.add("in"); });
  }

  // Optional link parameters: ?q=...&method=rl|fixed|single&budget=1..4&lang=en|ar&tab=steps|fruits|story|tries&solve=1
  function applyParams() {
    const p = new URLSearchParams(location.search);
    if (p.get("lang") === "ar" || p.get("lang") === "en") I18N.set(p.get("lang"));
    if (p.get("q")) { $("#question").value = p.get("q").slice(0, 1000); $("#question").dispatchEvent(new Event("input")); }
    const m = $(`input[name=method][value="${CSS.escape(p.get("method") || "")}"]`);
    if (m) m.checked = true;
    const b = parseInt(p.get("budget"), 10);
    if (b >= 1 && b <= 4) { $("#budget").value = b; $("#budgetVal").textContent = b; }
    const tab = p.get("tab") && $(`.tab[data-tab="${CSS.escape(p.get("tab"))}"]`);
    if (tab) tab.click();
    if (p.get("solve") === "1" && p.get("q")) {
      const go = () => (health && health.status === "ready" ? solve() : setTimeout(go, 500));
      go();
    }
  }

  Mascots.mount();
  I18N.apply();
  wire();
  pollHealth();
  loadResults();
  applyParams();
})();
