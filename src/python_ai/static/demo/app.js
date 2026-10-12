import { parseSSE } from './sse.js';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { mode: 'RAG', busy: false, uploading: false, hits: [], controller: null };
const titles = { workspace: '知识库工作台', changes: '优化前后对照', cases: '一步步验收' };
const storageKey = 'python-ai-demo-sessions-v1';
let sessions;
try { sessions = JSON.parse(localStorage.getItem(storageKey)) || {}; } catch { sessions = {}; }
if (typeof sessions !== 'object' || Array.isArray(sessions)) sessions = {};
let toastTimer;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function toast(message, error = false) {
  $('#toast').textContent = message;
  $('#toast').className = `toast${error ? ' error' : ''}`;
  $('#toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 5000);
}

function sessionId() {
  if (typeof sessions[state.mode] !== 'string' || !sessions[state.mode]) {
    sessions[state.mode] = `web-${crypto.randomUUID()}`;
    try { localStorage.setItem(storageKey, JSON.stringify(sessions)); } catch { /* Private mode. */ }
  }
  $('#session-label').textContent = `会话 ${sessions[state.mode].slice(-8)}`;
  return sessions[state.mode];
}

function showView(view) {
  if (!(view in titles)) view = 'workspace';
  $$('.view').forEach((element) => { element.hidden = element.id !== `view-${view}`; });
  $$('.nav-item[data-view]').forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle('active', active);
    if (active) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  $('#page-title').textContent = titles[view];
  history.replaceState(null, '', `#${view}`);
}

function setMode(mode) {
  if (state.busy) return;
  state.mode = mode;
  $$('.mode-switch button').forEach((button) => {
    button.classList.toggle('selected', button.dataset.mode === mode);
    button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  });
  $('#question').placeholder = mode === 'RAG'
    ? '问问这份资料，比如：网关超时是多少秒？' : '普通聊天不检索知识库，可以在这里测试多轮记忆。';
  sessionId();
  renderEvidence([]);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let body;
  try { body = await response.json(); } catch { throw new Error(`服务响应异常（HTTP ${response.status}）`); }
  if (!response.ok || body.code !== 0) throw new Error(body.message || `请求失败（${body.code}）`);
  return body.data;
}

async function refreshHealth() {
  try {
    const health = await api('/api/health');
    let ready = true;
    for (const service of ['postgres', 'qdrant', 'redis']) {
      const ok = Boolean(health[service]);
      ready = ready && ok;
      const tag = $(`#health-${service}`);
      tag.className = `health-tag ${ok ? 'good' : 'bad'}`;
      tag.querySelector('i').textContent = ok ? '就绪' : '不可用';
    }
    $('#rerank-status').textContent = `${health.hybridEnabled ? '混合检索' : '向量检索'} · ${health.rerankEnabled ? '重排开启' : '重排关闭'} · 追问改写${health.queryRewriteEnabled ? '开启' : '关闭'}`;
    $('#connection-text').textContent = ready ? '服务已连接' : '部分依赖不可用';
    $('#connection-dot').className = `dot${ready ? '' : ' bad'}`;
  } catch {
    $('#connection-text').textContent = '服务连接失败';
    $('#connection-dot').className = 'dot bad';
    for (const service of ['postgres', 'qdrant', 'redis']) {
      $(`#health-${service}`).className = 'health-tag bad';
      $(`#health-${service} i`).textContent = '未知';
    }
  }
}

async function refreshDocuments() {
  try {
    const docs = await api('/api/documents');
    $('#doc-count').textContent = String(docs.length);
    const container = $('#document-list');
    container.replaceChildren();
    if (!docs.length) container.append(node('p', 'empty-small', '知识库还是空的。先载入示例，开启第一次问答。'));
    for (const doc of docs) {
      const row = node('div', 'doc-row');
      row.dataset.docId = String(doc.id);
      row.append(node('span', 'doc-symbol', (doc.sourceType || 'DOC').toUpperCase()));
      const info = node('div', 'doc-info');
      const title = node('strong', '', doc.docName);
      title.title = doc.docName;
      const meta = node('div', 'doc-meta');
      meta.append(node('span', '', `ID ${doc.id} · ${doc.chunkCount} 片`));
      const statusNames = { DONE: '已入库', FAILED: '失败', PENDING: '待完成', PROCESSING: '处理中' };
      meta.append(node('span', `doc-status${doc.status === 'FAILED' ? ' failed' : ''}`, statusNames[doc.status] || doc.status));
      info.append(title, meta);
      if (doc.errorMsg) info.title = doc.errorMsg;
      const remove = node('button', 'delete-doc', '删除');
      remove.setAttribute('aria-label', `删除文档 ${doc.docName}`);
      remove.addEventListener('click', async () => {
        if (!confirm(`删除「${doc.docName}」及其检索向量？此操作不可撤销。`)) return;
        remove.disabled = true;
        try {
          await api(`/api/documents/${doc.id}`, { method: 'DELETE' });
          toast('文档及向量已删除。已有聊天内容仍保留，后续检索将使用剩余资料。');
          renderEvidence([]);
          await refreshDocuments();
        } catch (error) { toast(error.message, true); }
        finally { remove.disabled = false; }
      });
      row.append(info, remove);
      container.append(row);
    }
  } catch (error) {
    $('#document-list').replaceChildren(node('p', 'empty-small', `读取失败：${error.message}`));
    $('#doc-count').textContent = '—';
  }
}

async function upload(file) {
  if (state.uploading) return;
  state.uploading = true;
  $('#load-sample').disabled = true;
  $('#file-input').disabled = true;
  $('#upload-status').className = 'inline-status';
  $('#upload-status').textContent = `正在解析与入库：${file.name}…`;
  try {
    const form = new FormData();
    form.append('file', file);
    const result = await api('/api/documents', { method: 'POST', body: form });
    $('#upload-status').textContent = result.duplicated
      ? `内容已存在 · 文档 ID ${result.docId} · ${result.chunkCount} 片 · ${result.costMs} ms`
      : `入库成功 · 文档 ID ${result.docId} · ${result.chunkCount} 片 · ${result.costMs} ms`;
    toast(result.duplicated ? '相同内容已存在，本次跳过向量化。' : '文档已入库，现在可以向它提问。');
    await refreshDocuments();
  } catch (error) {
    $('#upload-status').textContent = `上传失败：${error.message}`;
    $('#upload-status').className = 'inline-status error';
    toast(error.message, true);
    await refreshDocuments();
  } finally {
    state.uploading = false;
    $('#load-sample').disabled = false;
    $('#file-input').disabled = false;
    $('#file-input').value = '';
  }
}

function renderEvidence(hits) {
  state.hits = hits;
  $('#hit-count').textContent = String(hits.length);
  const container = $('#evidence-list');
  container.replaceChildren();
  if (!hits.length) {
    const empty = node('div', 'evidence-empty');
    empty.append(node('span', '', '⌕'), node('h3', '', state.mode === 'CHAT' ? '普通聊天不检索资料' : '暂无检索依据'), node('p', '', state.mode === 'CHAT' ? '此模式直接与模型对话，可测试多轮记忆。' : '无命中时，系统会明确说明资料不足。'));
    container.append(empty);
  }
  hits.forEach((hit, index) => {
    const card = node('article', 'evidence-card');
    card.id = `source-${index + 1}`;
    const header = node('div', 'evidence-head');
    const name = node('span', '', `${hit.docName || '未命名文档'}${hit.sourcePage ? ` · 第 ${hit.sourcePage} 页` : ''}`);
    name.title = hit.docName || '';
    header.append(node('strong', '', String(index + 1).padStart(2, '0')), name);
    const scoreNames = { vector: '向量', bm25: 'BM25', rrf: '融合', rerank: '重排' };
    if (typeof hit.score === 'number') header.append(node('small', '', `${scoreNames[hit.scoreKind] || '分数'} ${hit.score.toFixed(4)}`));
    const content = hit.content || '';
    card.append(header, node('p', '', content.length > 180 ? `${content.slice(0, 180)}…` : content));
    const details = node('details');
    details.append(node('summary', '', `查看完整片段 · 文档 ${hit.docId} / 切片 ${hit.chunkId}`), node('p', '', content));
    card.append(details);
    container.append(card);
  });
}

function focusSource(index) {
  const card = $(`#source-${index}`);
  if (!card) return toast('该序号没有对应的当前检索片段。');
  card.querySelector('details').open = true;
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  card.animate([{ background: '#dceccb' }, { background: '#fbfdf7' }], { duration: 1300 });
}

// Create DOM nodes instead of injecting model text as HTML.
function inlineText(container, text, allowCitations) {
  for (const part of text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`|\[\d+\])/g)) {
    if (/^\*\*.*\*\*$/.test(part)) container.append(node('strong', '', part.slice(2, -2)));
    else if (/^`.*`$/.test(part)) container.append(node('code', '', part.slice(1, -1)));
    else if (/^\[\d+\]$/.test(part) && allowCitations) {
      const button = node('button', 'cite-button', part);
      button.type = 'button';
      button.setAttribute('aria-label', `查看来源 ${part}`);
      button.addEventListener('click', () => focusSource(Number(part.slice(1, -1))));
      container.append(button);
    } else container.append(document.createTextNode(part));
  }
}

function renderAnswer(container, text, allowCitations) {
  container.replaceChildren();
  const blocks = text.split(/```[^\n]*\n([\s\S]*?)```/g);
  blocks.forEach((block, index) => {
    if (index % 2) {
      const pre = node('pre');
      pre.append(node('code', '', block));
      container.append(pre);
    } else {
      for (const paragraph of block.split(/\n\s*\n/)) {
        if (!paragraph) continue;
        const p = node('p');
        paragraph.split('\n').forEach((line, lineIndex) => {
          if (lineIndex) p.append(document.createElement('br'));
          inlineText(p, line.replace(/^#{1,6} /, ''), allowCitations);
        });
        container.append(p);
      }
    }
  });
}

function message(role, text = '') {
  $('#welcome')?.remove();
  // A new response's citations must not navigate to an unrelated prior response.
  if (role === 'user') $$('.cite-button').forEach((button) => { button.disabled = true; });
  const row = node('div', 'message-row');
  const label = node('div', 'message-label');
  label.append(node('span', '', role === 'user' ? 'YOU / 你' : `知序 / ${state.mode === 'RAG' ? '知识库问答' : '普通聊天'}`));
  const timing = node('span');
  label.append(timing);
  const body = node('div', `message-body ${role}`, text);
  const query = node('p', 'query-info');
  query.hidden = true;
  row.append(label, body, query);
  $('#messages').append(row);
  return { body, timing, query };
}

function showQueryInfo(answer, info) {
  if (!info) return;
  const statusNames = { disabled: '改写关闭', no_history: '首轮使用原问题', unchanged: '保持原问题', rewritten: '已结合历史改写', fallback: '改写失败，使用原问题' };
  answer.query.textContent = `${statusNames[info.rewriteStatus] || '检索问题'}：${info.retrievalQuery}`;
  answer.query.hidden = false;
}

function showRunInfo(answer, result) {
  const line = node('p', 'query-info');
  line.textContent = result.memorySaved === false ? '本轮回答成功，但会话保存失败，请检查服务日志。' : '会话已保存';
  if (result.questionCategory) {
    const topic = result.questionCategory;
    line.append(document.createTextNode(topic.status === 'classified'
      ? ` · 模型主题：${topic.name}（${Math.round(topic.confidence * 100)}%）`
      : ' · 模型主题：未确定'));
  }
  if (result.traceUrl && /^https:\/\//.test(result.traceUrl)) {
    const link = node('a', '', '查看调用链 ↗');
    link.href = result.traceUrl;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    line.append(document.createTextNode(' · '), link);
  }
  answer.body.parentElement.append(line);
}

function setBusy(busy) {
  state.busy = busy;
  $('#send').disabled = busy;
  $('#new-session').disabled = busy;
  $('#stop').hidden = !busy;
  $$('.mode-switch button, [data-question]').forEach((button) => { button.disabled = busy; });
}

async function readStream(response, onEvent) {
  if (!response.ok || !response.headers.get('content-type')?.includes('text/event-stream')) {
    let body;
    try { body = await response.json(); } catch { /* Report HTTP status below. */ }
    throw new Error(body?.message || `流式请求失败（HTTP ${response.status}）`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      const parsed = parseSSE(buffer, done);
      buffer = parsed.remainder;
      for (const event of parsed.events) onEvent(event);
      if (done) break;
    }
  } finally { reader.releaseLock(); }
}

$('#chat-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (state.busy) return;
  const question = $('#question').value;
  if (!question.trim()) return toast('请输入问题，不能只包含空白字符。', true);
  if (question.length > 4000) return toast('问题不能超过 4000 个字符。', true);
  const mode = state.mode;
  const sid = sessionId();
  message('user', question);
  const answer = message('assistant', '正在检索与思考…');
  answer.body.classList.add('loading');
  $('#question').value = '';
  renderEvidence([]);
  $('#request-status').textContent = '正在处理，请稍候…';
  state.controller = new AbortController();
  setBusy(true);
  let text = '', completed = false, scheduled = false, repaintId = 0;
  const paint = () => {
    scheduled = false;
    answer.body.classList.remove('loading');
    renderAnswer(answer.body, text, mode === 'RAG');
    $('#messages').scrollTop = $('#messages').scrollHeight;
  };
  try {
    const options = { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: question, sessionId: sid }), signal: state.controller.signal };
    if ($('#stream-toggle').checked) {
      const response = await fetch(`/api/chat/stream?mode=${mode}`, options);
      await readStream(response, ({ type, data }) => {
        if (type === 'retrieval') renderEvidence(JSON.parse(data));
        if (type === 'message') {
          text += data;
          if (!scheduled) { scheduled = true; repaintId = requestAnimationFrame(paint); }
        }
        if (type === 'error') throw new Error(data);
        if (type === 'done') {
          const result = JSON.parse(data);
          completed = true;
          answer.timing.textContent = `${result.costMs} ms`;
          showQueryInfo(answer, result.queryInfo);
          showRunInfo(answer, result);
        }
      });
      if (!completed) throw new Error('连接提前结束，未收到完成事件；部分回答未作为完整一轮保存。');
    } else {
      const result = await api(`/api/chat?mode=${mode}`, options);
      text = result.answer;
      renderEvidence(result.hits || []);
      answer.timing.textContent = `${result.costMs} ms`;
      showQueryInfo(answer, result.queryInfo);
      showRunInfo(answer, result);
      completed = true;
    }
    cancelAnimationFrame(repaintId);
    paint();
    $('#request-status').textContent = `已完成 · ${mode === 'RAG' ? `${state.hits.length} 个来源` : '普通聊天'} · ${answer.timing.textContent}`;
  } catch (error) {
    cancelAnimationFrame(repaintId);
    const stopped = error.name === 'AbortError';
    const detail = stopped ? '已停止生成。' : `请求失败：${error.message}`;
    // Preserve partial output with an explicit failure marker.
    if (text) paint(); else answer.body.replaceChildren();
    answer.body.classList.remove('loading');
    answer.body.append(node('p', 'error', detail));
    answer.timing.textContent = stopped ? '已停止' : '未完成';
    $('#request-status').textContent = detail;
    if (!stopped) toast(error.message, true);
  } finally {
    // Prevent queued streaming repaint from erasing an error label.
    scheduled = false;
    cancelAnimationFrame(repaintId);
    state.controller = null;
    setBusy(false);
  }
});

$('#question').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $('#chat-form').requestSubmit();
  }
});
$('#stop').addEventListener('click', () => state.controller?.abort());
$('#new-session').addEventListener('click', () => {
  if (state.busy) return;
  delete sessions[state.mode];
  sessionId();
  const welcome = node('div', 'welcome', '新会话已准备好。可以开始一个新的问题。');
  welcome.id = 'welcome';
  $('#messages').replaceChildren(welcome);
  renderEvidence([]);
  $('#request-status').textContent = '新会话 · 本页已清空聊天显示';
  toast('已创建新会话，后续请求不携带旧会话历史。');
});
$$('.mode-switch button').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
$$('[data-question]').forEach((button) => button.addEventListener('click', () => {
  if (state.busy) return;
  showView('workspace');
  if (button.dataset.mode) setMode(button.dataset.mode);
  $('#question').value = button.dataset.question;
  $('#question').focus();
  toast('问题已填入，点击发送即可体验。');
}));
$$('.nav-item[data-view]').forEach((button) => button.addEventListener('click', () => showView(button.dataset.view)));
$$('[data-go]').forEach((button) => button.addEventListener('click', () => showView(button.dataset.go)));
$('#refresh').addEventListener('click', () => Promise.all([refreshHealth(), refreshDocuments()]));
$('#refresh-docs').addEventListener('click', refreshDocuments);
$('#file-input').addEventListener('change', (event) => { if (event.target.files[0]) upload(event.target.files[0]); });
$('#load-sample').addEventListener('click', async () => {
  if (state.uploading) return;
  try {
    const response = await fetch('/demo/sample');
    if (!response.ok) throw new Error('示例文档加载失败');
    await upload(new File([await response.blob()], 'star-link-spec.md', { type: 'text/markdown' }));
  } catch (error) { toast(error.message, true); }
});
for (const type of ['dragenter', 'dragover']) $('#dropzone').addEventListener(type, (event) => {
  event.preventDefault();
  $('#dropzone').classList.add('dragover');
});
for (const type of ['dragleave', 'drop']) $('#dropzone').addEventListener(type, (event) => {
  event.preventDefault();
  $('#dropzone').classList.remove('dragover');
  if (type === 'drop' && event.dataTransfer.files[0]) upload(event.dataTransfer.files[0]);
});

showView(location.hash.slice(1));
sessionId();
Promise.all([refreshHealth(), refreshDocuments()]);
