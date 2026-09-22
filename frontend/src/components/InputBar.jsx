import { useRef, useEffect, useCallback } from 'react';

export default function InputBar({ value, onChange, onSubmit, disabled }) {
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, [value]);

  const handleKey = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSubmit();
    }
  }, [disabled, value, onSubmit]);

  return (
    <div className="input-bar">
      <div className="input-wrap">
        <textarea
          ref={textareaRef}
          className="question-input"
          placeholder="Ask a question about your documents..."
          rows={1}
          maxLength={1000}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKey}
          disabled={disabled}
        />
        <button
          className="send-btn"
          aria-label="Send"
          disabled={disabled || !value.trim()}
          onClick={onSubmit}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
          </svg>
        </button>
      </div>
      <p className="input-hint">Press Enter to send, Shift+Enter for new line</p>
    </div>
  );
}
