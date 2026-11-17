/* Original cartoon mascots drawn as inline SVG (no external art). */
(function () {
  const robo = `
<svg viewBox="0 0 120 120" class="mascot-svg" role="img" aria-label="Robo">
  <line x1="60" y1="16" x2="60" y2="30" stroke="#9fb4ff" stroke-width="4" stroke-linecap="round"/>
  <circle class="antenna" cx="60" cy="13" r="6.5" fill="#ffd166"/>
  <rect x="11" y="50" width="12" height="22" rx="6" fill="#2f9fd0"/>
  <rect x="97" y="50" width="12" height="22" rx="6" fill="#2f9fd0"/>
  <rect x="20" y="29" width="80" height="62" rx="24" fill="#4cc9f0"/>
  <rect x="20" y="29" width="80" height="20" rx="10" fill="#7ad8f5" opacity=".45"/>
  <rect x="31" y="41" width="58" height="36" rx="15" fill="#13264d"/>
  <g class="eyes">
    <ellipse cx="47" cy="57" rx="6" ry="7" fill="#7df9ff"/>
    <ellipse cx="73" cy="57" rx="6" ry="7" fill="#7df9ff"/>
    <circle cx="49" cy="55" r="2" fill="#fff"/><circle cx="75" cy="55" r="2" fill="#fff"/>
  </g>
  <path d="M51 68 Q60 74 69 68" stroke="#7df9ff" stroke-width="3" fill="none" stroke-linecap="round"/>
  <circle cx="33" cy="80" r="4" fill="#ff8fab" opacity=".75"/><circle cx="87" cy="80" r="4" fill="#ff8fab" opacity=".75"/>
  <rect x="38" y="90" width="44" height="22" rx="9" fill="#2f9fd0"/>
  <text x="60" y="106" text-anchor="middle" font-size="13" font-weight="800" fill="#13264d" font-family="Arial">+ ×</text>
</svg>`;

  const hoot = `
<svg viewBox="0 0 120 120" class="mascot-svg" role="img" aria-label="Hoot">
  <path d="M27 42 L33 14 L50 33 Z" fill="#7b5ce6"/>
  <path d="M93 42 L87 14 L70 33 Z" fill="#7b5ce6"/>
  <ellipse cx="60" cy="68" rx="39" ry="43" fill="#a685ff"/>
  <path d="M24 70 Q14 86 28 98 Q30 84 34 76 Z" fill="#8e6cf0"/>
  <path d="M96 70 Q106 86 92 98 Q90 84 86 76 Z" fill="#8e6cf0"/>
  <ellipse cx="60" cy="84" rx="24" ry="23" fill="#e3d9ff"/>
  <path d="M48 80 q4 4 8 0 M58 88 q4 4 8 0 M50 96 q4 4 8 0" stroke="#b9a6f5" stroke-width="2" fill="none"/>
  <circle cx="44" cy="54" r="15" fill="#fff"/><circle cx="76" cy="54" r="15" fill="#fff"/>
  <circle cx="44" cy="54" r="16" fill="none" stroke="#ffd166" stroke-width="4"/>
  <circle cx="76" cy="54" r="16" fill="none" stroke="#ffd166" stroke-width="4"/>
  <line x1="60" y1="53" x2="60" y2="53" stroke="#ffd166" stroke-width="4"/>
  <g class="eyes"><circle cx="46" cy="56" r="6.5" fill="#1b1546"/><circle cx="78" cy="56" r="6.5" fill="#1b1546"/>
    <circle cx="48" cy="54" r="2" fill="#fff"/><circle cx="80" cy="54" r="2" fill="#fff"/></g>
  <path d="M54 66 L66 66 L60 76 Z" fill="#ffb347"/>
  <ellipse cx="48" cy="111" rx="7" ry="4" fill="#ffb347"/><ellipse cx="72" cy="111" rx="7" ry="4" fill="#ffb347"/>
  <g class="glass">
    <line x1="104" y1="101" x2="114" y2="113" stroke="#c98a3a" stroke-width="5" stroke-linecap="round"/>
    <circle cx="97" cy="93" r="11" fill="rgba(255,255,255,.28)" stroke="#ffd166" stroke-width="4"/>
  </g>
</svg>`;

  const captain = `
<svg viewBox="0 0 120 120" class="mascot-svg" role="img" aria-label="Captain Q">
  <path d="M24 56 L30 20 L52 42 Z" fill="#ff8c42"/><path d="M31 48 L33 29 L45 41 Z" fill="#ffd6a5"/>
  <path d="M96 56 L90 20 L68 42 Z" fill="#ff8c42"/><path d="M89 48 L87 29 L75 41 Z" fill="#ffd6a5"/>
  <ellipse cx="60" cy="70" rx="38" ry="34" fill="#ff8c42"/>
  <path d="M24 74 Q60 112 96 74 Q84 100 60 102 Q36 100 24 74 Z" fill="#fff3e0"/>
  <ellipse cx="60" cy="86" rx="20" ry="14" fill="#fff3e0"/>
  <g class="eyes"><ellipse cx="46" cy="68" rx="5" ry="6" fill="#1b1546"/><ellipse cx="74" cy="68" rx="5" ry="6" fill="#1b1546"/>
    <circle cx="47.5" cy="66" r="1.8" fill="#fff"/><circle cx="75.5" cy="66" r="1.8" fill="#fff"/></g>
  <ellipse cx="60" cy="80" rx="5" ry="3.8" fill="#1b1546"/>
  <path d="M52 86 Q60 93 68 86" stroke="#1b1546" stroke-width="2.6" fill="none" stroke-linecap="round"/>
  <circle cx="36" cy="82" r="4" fill="#ff6b8b" opacity=".55"/><circle cx="84" cy="82" r="4" fill="#ff6b8b" opacity=".55"/>
  <path d="M30 40 Q60 18 90 40 L86 46 Q60 34 34 46 Z" fill="#1b1546"/>
  <rect x="26" y="42" width="68" height="8" rx="4" fill="#2a2168"/>
  <circle cx="60" cy="33" r="9" fill="#ffd166"/>
  <text x="60" y="37.5" text-anchor="middle" font-size="12" font-weight="900" fill="#1b1546" font-family="Arial">Q</text>
</svg>`;

  const ART = { robo, hoot, captain };
  window.Mascots = {
    svg: (name) => ART[name] || "",
    mount(root = document) {
      root.querySelectorAll("[data-mascot]").forEach((el) => {
        if (!el.dataset.mounted) { el.innerHTML = ART[el.dataset.mascot] || ""; el.dataset.mounted = "1"; }
      });
    },
  };
})();
