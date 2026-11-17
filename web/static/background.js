/* Floating math symbols on a canvas behind the page. Calm, low-contrast, pauses when hidden
   and stays still for people who prefer reduced motion. */
(function () {
  const canvas = document.getElementById("bg");
  const ctx = canvas.getContext("2d");
  const SYMBOLS = ["+", "−", "×", "÷", "=", "π", "√", "∑", "∞", "%", "½", "¾", "x²", "≈", "≠", "<", ">", "∫", "Δ", "θ",
                   "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "🍎", "⭐", "🎈"];
  const COLORS = ["#ffd166", "#ff6b8b", "#3ddc97", "#4cc9f0", "#b388ff", "#ff9f43"];
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let W = 0, H = 0, dpr = 1, items = [], last = 0, running = true;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const count = Math.round(Math.min(70, Math.max(26, (W * H) / 26000)));
    while (items.length < count) items.push(make(true));
    items.length = count;
  }

  function make(anywhere) {
    const size = 14 + Math.random() * 30;
    return {
      ch: SYMBOLS[(Math.random() * SYMBOLS.length) | 0],
      x: Math.random() * W,
      y: anywhere ? Math.random() * H : H + size,
      size,
      vy: 8 + Math.random() * 22,          // px per second, upward
      vx: (Math.random() - 0.5) * 8,
      rot: Math.random() * Math.PI,
      vr: (Math.random() - 0.5) * 0.6,
      alpha: 0.10 + Math.random() * 0.18,
      color: COLORS[(Math.random() * COLORS.length) | 0],
      wobble: Math.random() * Math.PI * 2,
    };
  }

  function draw(t) {
    if (!running) return;
    const dt = Math.min(0.05, (t - last) / 1000 || 0);
    last = t;
    ctx.clearRect(0, 0, W, H);
    for (const p of items) {
      if (!reduce) {
        p.y -= p.vy * dt;
        p.wobble += dt;
        p.x += (p.vx + Math.sin(p.wobble) * 6) * dt;
        p.rot += p.vr * dt;
        if (p.y < -60) Object.assign(p, make(false));
      }
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle = p.color;
      ctx.font = `800 ${p.size}px "Baloo Bhaijaan 2", "Segoe UI", sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(p.ch, 0, 0);
      ctx.restore();
    }
    if (!reduce) requestAnimationFrame(draw);
  }

  window.addEventListener("resize", () => { resize(); if (reduce) draw(performance.now()); });
  document.addEventListener("visibilitychange", () => {
    running = !document.hidden;
    if (running && !reduce) { last = performance.now(); requestAnimationFrame(draw); }
  });
  resize();
  requestAnimationFrame(draw);
})();
