const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(method, path, body = null, isFormData = false) {
  const options = { method, headers: {} };
  if (body && isFormData) {
    options.body = body;
  } else if (body) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const error = new Error(err.detail || 'Request failed');
    error.status = res.status;
    throw error;
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  health:          ()          => request('GET',    '/api/health'),
  listDocuments:   ()          => request('GET',    '/api/documents'),
  deleteDocument:  (id)        => request('DELETE', `/api/documents/${id}`),
  uploadDocument:  (formData)  => request('POST',   '/api/documents', formData, true),
  query:           (question)  => request('POST',   '/api/query', { question }),
};
