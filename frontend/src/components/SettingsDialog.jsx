import { useState } from "react";
import { Modal } from "./common.jsx";
import { SEEDS } from "../lib/prefs.js";

const THEME_MODES = [
  ["auto", "跟随系统", "brightness_auto"],
  ["light", "浅色", "light_mode"],
  ["dark", "深色", "dark_mode"],
];

const SHORTCUTS = [
  ["Enter", "发送"],
  ["Shift + Enter", "换行"],
  ["Alt + 1…4", "切换模式：问答 / 分析 / 上下文 / 教学"],
  ["Ctrl + Shift + O", "发起新对话"],
  ["/", "聚焦输入框"],
];

export default function SettingsDialog({ open, onClose, prefs, onPrefs, onClearAll, conversationCount }) {
  const [confirmClear, setConfirmClear] = useState(false);
  return (
    <Modal open={open} onClose={onClose} className="settings-dialog">
      <div className="dialog-head">
        <h5 className="max">设置</h5>
        <button className="circle transparent" onClick={onClose} aria-label="关闭">
          <i>close</i>
        </button>
      </div>

      <h6 className="sub-title">外观</h6>
      <nav className="seg-buttons">
        {THEME_MODES.map(([k, name, icon]) => (
          <button key={k} className={prefs.themeMode === k ? "on" : ""} onClick={() => onPrefs({ themeMode: k })}>
            <i>{icon}</i>
            <span>{name}</span>
          </button>
        ))}
      </nav>

      <h6 className="sub-title">主题色</h6>
      <div className="swatches">
        {SEEDS.map((s) => (
          <button
            key={s.color}
            className={`swatch ${prefs.seed === s.color ? "on" : ""}`}
            style={{ "--sw": s.color }}
            onClick={() => onPrefs({ seed: s.color })}
            title={s.name}
            aria-label={s.name}
          >
            {prefs.seed === s.color ? <i>check</i> : null}
          </button>
        ))}
        <label className="swatch custom" title="自定义颜色">
          <i>colorize</i>
          <input type="color" value={prefs.seed} onChange={(e) => onPrefs({ seed: e.target.value })} />
        </label>
      </div>

      <h6 className="sub-title">快捷键</h6>
      <div className="shortcuts">
        {SHORTCUTS.map(([k, v]) => (
          <div key={k} className="shortcut">
            <kbd>{k}</kbd>
            <span>{v}</span>
          </div>
        ))}
      </div>

      <h6 className="sub-title">本地数据</h6>
      <p className="muted small-text">
        对话记录和上下文只保存在本浏览器（localStorage），不会发送给后端，也不会写入知识库。当前共 {conversationCount} 个对话。
      </p>
      {confirmClear ? (
        <nav className="confirm-row">
          <span className="max">确定删除全部对话记录？此操作不可撤销。</span>
          <button className="border small" onClick={() => setConfirmClear(false)}>
            取消
          </button>
          <button
            className="error small"
            onClick={() => {
              onClearAll();
              setConfirmClear(false);
            }}
          >
            删除
          </button>
        </nav>
      ) : (
        <button className="border error-text" disabled={!conversationCount} onClick={() => setConfirmClear(true)}>
          <i>delete_sweep</i>
          <span>清除全部对话</span>
        </button>
      )}
    </Modal>
  );
}
