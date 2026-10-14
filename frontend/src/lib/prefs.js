// Per-browser UI preferences (theme, last mode). Storage may be unavailable
// (private mode, blocked site data), so every access is guarded.

const KEY = "cognitive-ai.prefs.v1";

export const SEEDS = [
  { name: "湛蓝", color: "#3f6ad8" },
  { name: "紫罗兰", color: "#6750a4" },
  { name: "青绿", color: "#00796b" },
  { name: "琥珀", color: "#b36b00" },
  { name: "玫红", color: "#b0306a" },
  { name: "石墨", color: "#5f6368" },
];

export const DEFAULT_PREFS = { themeMode: "auto", seed: SEEDS[0].color, mode: "ask", sidebar: true };

export function safeStorage() {
  try {
    const s = globalThis.localStorage;
    if (!s) return null;
    const probe = "__cognitive_ai_probe__";
    s.setItem(probe, "1");
    s.removeItem(probe);
    return s;
  } catch {
    return null;
  }
}

export function loadPrefs(storage = safeStorage()) {
  try {
    const raw = storage?.getItem(KEY);
    return raw ? { ...DEFAULT_PREFS, ...JSON.parse(raw) } : { ...DEFAULT_PREFS };
  } catch {
    return { ...DEFAULT_PREFS };
  }
}

export function savePrefs(prefs, storage = safeStorage()) {
  try {
    storage?.setItem(KEY, JSON.stringify(prefs));
  } catch {
    /* preferences are a convenience only */
  }
}
