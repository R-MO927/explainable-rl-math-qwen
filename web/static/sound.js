/* Playful sound effects synthesized with the Web Audio API (no audio files).
   Sfx.play(name); Sfx.toggle() mutes/unmutes and is remembered per browser. */
(function () {
  let ctx = null, master = null, muted = false;
  try { muted = localStorage.getItem("mm-muted") === "1"; } catch (e) { /* storage unavailable */ }

  function audio() {
    if (!ctx) {
      const C = window.AudioContext || window.webkitAudioContext;
      if (!C) return null;
      ctx = new C();
      const comp = ctx.createDynamicsCompressor();
      master = ctx.createGain();
      master.gain.value = 0.55;
      master.connect(comp).connect(ctx.destination);
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  function tone({ f = 440, f2 = null, type = "sine", at = 0, d = 0.15, v = 0.15, attack = 0.006 }) {
    const a = audio();
    if (!a || muted) return;
    const t0 = a.currentTime + at;
    const o = a.createOscillator(), g = a.createGain();
    o.type = type;
    o.frequency.setValueAtTime(f, t0);
    if (f2) o.frequency.exponentialRampToValueAtTime(f2, t0 + d);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(v, t0 + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + d);
    o.connect(g).connect(master);
    o.start(t0);
    o.stop(t0 + d + 0.03);
  }

  function noise({ at = 0, d = 0.3, v = 0.12, from = 400, to = 2400, q = 1.2 }) {
    const a = audio();
    if (!a || muted) return;
    const t0 = a.currentTime + at;
    const buf = a.createBuffer(1, Math.ceil(a.sampleRate * d), a.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
    const src = a.createBufferSource(), bp = a.createBiquadFilter(), g = a.createGain();
    src.buffer = buf;
    bp.type = "bandpass"; bp.Q.value = q;
    bp.frequency.setValueAtTime(from, t0);
    bp.frequency.exponentialRampToValueAtTime(to, t0 + d);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(v, t0 + d * 0.3);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + d);
    src.connect(bp).connect(g).connect(master);
    src.start(t0);
    src.stop(t0 + d + 0.02);
  }

  const notes = (list, { type = "triangle", gap = 0.09, d = 0.18, v = 0.14, at = 0 } = {}) =>
    list.forEach((f, i) => tone({ f, type, at: at + i * gap, d, v }));

  const SOUNDS = {
    click: () => tone({ f: 520, f2: 880, type: "triangle", d: 0.08, v: 0.12 }),
    pop: () => { tone({ f: 380, f2: 1100, type: "sine", d: 0.09, v: 0.16 }); tone({ f: 1500, type: "sine", at: 0.05, d: 0.05, v: 0.05 }); },
    tick: () => tone({ f: 1250, type: "sine", d: 0.04, v: 0.06 }),
    select: () => notes([660, 990], { gap: 0.06, d: 0.12, v: 0.12 }),
    tab: () => { tone({ f: 740, f2: 990, type: "triangle", d: 0.07, v: 0.1 }); tone({ f: 1480, type: "sine", at: 0.04, d: 0.05, v: 0.04 }); },
    swoosh: () => { noise({ d: 0.35, v: 0.14, from: 300, to: 3200 }); notes([523, 784], { at: 0.12, gap: 0.08, d: 0.14, v: 0.1 }); },
    launch: () => {
      noise({ d: 0.7, v: 0.1, from: 200, to: 1800, q: 0.8 });
      tone({ f: 180, f2: 1200, type: "triangle", d: 0.6, v: 0.13 });
      notes([1568, 2093], { type: "sine", at: 0.55, gap: 0.07, d: 0.12, v: 0.07 });
    },
    blip: () => tone({ f: 620, f2: 700, type: "sine", d: 0.1, v: 0.045 }),
    success: () => {
      notes([523, 659, 784, 1047], { gap: 0.1, d: 0.22, v: 0.14 });
      notes([1319, 1568, 2093], { type: "sine", at: 0.42, gap: 0.06, d: 0.18, v: 0.06 });
    },
    sparkle: () => { for (let i = 0; i < 7; i++) tone({ f: 1600 + Math.random() * 1800, type: "sine", at: 0.05 + i * 0.06, d: 0.09, v: 0.04 }); },
    uhoh: () => { tone({ f: 494, f2: 470, type: "triangle", d: 0.2, v: 0.13 }); tone({ f: 392, f2: 330, type: "triangle", at: 0.22, d: 0.32, v: 0.13 }); },
    error: () => { tone({ f: 320, f2: 110, type: "sine", d: 0.4, v: 0.18 }); tone({ f: 640, f2: 220, type: "triangle", d: 0.25, v: 0.05 }); },
    robo: () => notes([880, 660, 990, 1320], { type: "square", gap: 0.07, d: 0.06, v: 0.05 }),
    hoot: () => { tone({ f: 420, f2: 360, type: "sine", d: 0.22, v: 0.18, attack: 0.04 }); tone({ f: 420, f2: 340, type: "sine", at: 0.3, d: 0.3, v: 0.18, attack: 0.04 }); },
    captain: () => { tone({ f: 1400, f2: 2200, type: "sine", d: 0.14, v: 0.1 }); tone({ f: 2200, f2: 1600, type: "sine", at: 0.15, d: 0.22, v: 0.1 }); },
  };

  window.Sfx = {
    play(name) { try { (SOUNDS[name] || (() => {}))(); } catch (e) { /* audio unavailable */ } },
    get muted() { return muted; },
    toggle() {
      muted = !muted;
      try { localStorage.setItem("mm-muted", muted ? "1" : "0"); } catch (e) { /* ignore */ }
      if (!muted) SOUNDS.select();
      return muted;
    },
  };
})();
