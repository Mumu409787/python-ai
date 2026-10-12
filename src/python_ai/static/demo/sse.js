// Separate framing from rendering so split UTF-8/SSE frames can be tested offline.
export function parseSSE(buffer, flush = false) {
  const events = [];
  let match;
  while ((match = /\r?\n\r?\n/.exec(buffer))) {
    const frame = buffer.slice(0, match.index);
    buffer = buffer.slice(match.index + match[0].length);
    const event = parseFrame(frame);
    if (event) events.push(event);
  }
  if (flush && buffer.trim()) {
    const event = parseFrame(buffer);
    if (event) events.push(event);
    buffer = '';
  }
  return { events, remainder: buffer };
}

function parseFrame(frame) {
  let type = 'message';
  const data = [];
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith('event:')) type = line.slice(6).trim();
    if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
  }
  return data.length ? { type, data: data.join('\n') } : null;
}
