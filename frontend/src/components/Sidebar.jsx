import { useState, useCallback } from 'react';
import { api } from '../api/client';

const ACCEPT = '.pdf,.txt,.md,.docx';

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function TypeBadge({ type }) {
  return <span className={`type-badge type-${type.toLowerCase()}`}>{type}</span>;
}

function DocItem({ doc, onDelete, activeDocId, onSelectDoc }) {
  return (
    <div
      className={`doc-item ${activeDocId === doc.doc_id ? 'active' : ''}`}
      onClick={() => onSelectDoc?.(doc)}
      style={{ cursor: 'pointer' }}
    >
      <TypeBadge type={doc.file_type} />
      <div className="doc-info">
        <p className="doc-name" title={doc.filename}>{doc.filename}</p>
        <p className="doc-size">{formatBytes(doc.size_bytes)}</p>
      </div>
      <button
        className="doc-delete"
        aria-label={`Remove ${doc.filename}`}
        onClick={(e) => {
          e.stopPropagation();
          onDelete(doc.doc_id);
        }}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}

export default function Sidebar({ docs, onDocsChange, onToast, modelReady, activeDocId, onSelectDoc }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  const handleFile = useCallback(async (file) => {
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'txt', 'md', 'docx'].includes(ext)) {
      onToast(`Unsupported file type: .${ext}`, 'error');
      return;
    }
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const record = await api.uploadDocument(fd);
      onDocsChange((prev) => [...prev, record]);
      onToast(`"${file.name}" uploaded successfully`);
    } catch (err) {
      onToast(err.message, 'error');
    } finally {
      setUploading(false);
    }
  }, [onDocsChange, onToast]);

  const handleDelete = useCallback(async (id) => {
    try {
      await api.deleteDocument(id);
      onDocsChange((prev) => prev.filter((d) => d.doc_id !== id));
      onToast('Document removed');
    } catch (err) {
      if (err.status === 404) {
        onDocsChange((prev) => prev.filter((d) => d.doc_id !== id));
        onToast('Document already removed');
      } else {
        onToast(err.message, 'error');
      }
    }
  }, [onDocsChange, onToast]);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="logo">
          <div className="logo-icon">R</div>
          <span className="logo-text">Corrective RAG</span>
        </div>
        <div
          className={`status-dot ${modelReady === true ? 'ready' : modelReady === false ? 'offline' : ''}`}
          title={modelReady === true ? 'Model ready' : modelReady === false ? 'Model offline' : 'Checking...'}
        />
      </div>

      <div className="sidebar-upload">
        <p className="section-label">Documents</p>

        <div
          className={`drop-zone${dragging ? ' drag-over' : ''}`}
          role="button"
          tabIndex={0}
          aria-label="Upload document"
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => document.getElementById('file-input').click()}
          onKeyDown={(e) => e.key === 'Enter' && document.getElementById('file-input').click()}
        >
          <div className="drop-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
          </div>
          {uploading ? (
            <p className="drop-primary">Uploading...</p>
          ) : (
            <>
              <p className="drop-primary">Drop a file here</p>
              <p className="drop-secondary">or click to browse</p>
              <p className="drop-hint">PDF, TXT, MD, DOCX</p>
            </>
          )}
          <input
            id="file-input"
            type="file"
            accept={ACCEPT}
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
            onClick={(e) => { e.target.value = ''; }}
          />
        </div>
      </div>

      <div className="doc-list-wrap">
        {docs.length === 0 ? (
          <div className="doc-empty">
            <p>No documents yet.</p>
            <p>Upload a file to get started.</p>
          </div>
        ) : (
          <div className="doc-list">
            {docs.map((doc) => (
              <DocItem 
                key={doc.doc_id} 
                doc={doc} 
                onDelete={handleDelete}
                activeDocId={activeDocId}
                onSelectDoc={onSelectDoc}
              />
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
