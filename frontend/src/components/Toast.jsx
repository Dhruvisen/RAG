import { useEffect, useCallback, useRef } from 'react';

export default function Toast({ message, type, onDismiss }) {
  const timerRef = useRef(null);

  useEffect(() => {
    if (!message) return;
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(onDismiss, 3500);
    return () => clearTimeout(timerRef.current);
  }, [message, onDismiss]);

  if (!message) return null;

  return (
    <div className={`toast show${type === 'error' ? ' error' : ''}`} role="alert">
      {message}
    </div>
  );
}
