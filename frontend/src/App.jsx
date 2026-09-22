import { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import InputBar from './components/InputBar';
import Toast from './components/Toast';
import { api } from './api/client';

let msgId = 0;
const nextId = () => ++msgId;

export default function App() {
  const [docs, setDocs]           = useState([]);
  const [messages, setMessages]   = useState([]);
  const [question, setQuestion]   = useState('');
  const [thinking, setThinking]   = useState(false);
  const [modelReady, setModelReady] = useState(null);  // null=checking, true, false
  const [previewDoc, setPreviewDoc] = useState(null);
  const [toast, setToast]         = useState({ message: '', type: '' });

  const showToast = useCallback((message, type = 'info') => {
    setToast({ message, type });
  }, []);

  const dismissToast = useCallback(() => {
    setToast({ message: '', type: '' });
  }, []);

  // Load documents and health on mount
  useEffect(() => {
    api.listDocuments()
      .then(setDocs)
      .catch(() => showToast('Could not load documents', 'error'));

    api.health()
      .then((h) => setModelReady(h.model_ready))
      .catch(() => setModelReady(false));
  }, []);

  const handleSubmit = useCallback(async () => {
    const q = question.trim();
    if (!q || thinking) return;

    setQuestion('');
    setMessages((prev) => [...prev, { id: nextId(), role: 'user', text: q }]);
    setThinking(true);

    try {
      const { answer } = await api.query(q);
      setMessages((prev) => [...prev, { id: nextId(), role: 'assistant', text: answer }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: 'assistant', text: `Error: ${err.message}` },
      ]);
    } finally {
      setThinking(false);
    }
  }, [question, thinking]);

  return (
    <div className="layout">
      <Sidebar
        docs={docs}
        onDocsChange={setDocs}
        onToast={showToast}
        modelReady={modelReady}
        activeDocId={previewDoc?.doc_id}
        onSelectDoc={(doc) => setPreviewDoc(doc)}
      />

      {previewDoc && (
        <div className="preview-panel">
          <div className="preview-header">
            <h3>{previewDoc.filename}</h3>
            <button className="close-preview" onClick={() => setPreviewDoc(null)}>✕</button>
          </div>
          <iframe 
            src={`/api/documents/${previewDoc.doc_id}/content`} 
            className="preview-iframe" 
            title="Document Preview"
          />
        </div>
      )}

      <main className="chat-area" style={{ display: previewDoc ? 'none' : 'flex' }}>
        <ChatArea messages={messages} thinking={thinking} />
        <InputBar
          value={question}
          onChange={setQuestion}
          onSubmit={handleSubmit}
          disabled={thinking}
        />
      </main>

      <Toast
        message={toast.message}
        type={toast.type}
        onDismiss={dismissToast}
      />
    </div>
  );
}
