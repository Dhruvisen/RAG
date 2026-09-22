import { useEffect, useRef } from 'react';

function ThinkingBubble() {
  return (
    <div className="msg-row assistant">
      <div className="bubble thinking-bubble">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </div>
    </div>
  );
}

function Message({ msg }) {
  return (
    <div className={`msg-row ${msg.role}`}>
      <div className="bubble">
        {msg.text}
      </div>
    </div>
  );
}

export default function ChatArea({ messages, thinking }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, thinking]);

  return (
    <div className="chat-messages">
      {messages.length === 0 && !thinking ? (
        <div className="welcome">
          <h1 className="welcome-title">Ask anything</h1>
          <p className="welcome-sub">
            Upload a document in the sidebar, then ask a question about its contents.
          </p>
        </div>
      ) : (
        <>
          {messages.map((msg) => (
            <Message key={msg.id} msg={msg} />
          ))}
          {thinking && <ThinkingBubble />}
          <div ref={bottomRef} />
        </>
      )}
    </div>
  );
}
