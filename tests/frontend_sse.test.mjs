import test from 'node:test';
import assert from 'node:assert/strict';
import { parseSSE } from '../src/python_ai/static/demo/sse.js';

test('CRLF frames survive arbitrary network chunk boundaries', () => {
  const stream = 'event: retrieval\r\ndata: []\r\n\r\nevent: message\r\ndata: 你好\r\n\r\nevent: done\r\ndata: {"costMs":12}\r\n\r\n';
  for (let offset = 1; offset < stream.length; offset++) {
    const first = parseSSE(stream.slice(0, offset));
    const second = parseSSE(first.remainder + stream.slice(offset));
    assert.deepEqual([...first.events, ...second.events], [
      { type: 'retrieval', data: '[]' }, { type: 'message', data: '你好' },
      { type: 'done', data: '{"costMs":12}' },
    ]);
    assert.equal(second.remainder, '');
  }
});

test('comments, multiline data and significant leading spaces', () => {
  assert.deepEqual(parseSSE(': ping\n\nevent: message\ndata:   code\ndata: next\n\n').events,
    [{ type: 'message', data: '  code\nnext' }]);
});

test('empty data is a message, incomplete frame stays buffered', () => {
  assert.deepEqual(parseSSE('data:\n\ndata: next'), {
    events: [{ type: 'message', data: '' }], remainder: 'data: next',
  });
  assert.deepEqual(parseSSE('event: error\ndata: 模型超时', true), {
    events: [{ type: 'error', data: '模型超时' }], remainder: '',
  });
});

test('UTF-8 characters survive split byte chunks before framing', () => {
  const bytes = new TextEncoder().encode('event: message\ndata: 中文回答\n\n');
  const decoder = new TextDecoder();
  let buffer = '', events = [];
  for (const byte of bytes) {
    const result = parseSSE(buffer + decoder.decode(new Uint8Array([byte]), { stream: true }));
    events.push(...result.events);
    buffer = result.remainder;
  }
  assert.deepEqual(events, [{ type: 'message', data: '中文回答' }]);
  assert.equal(buffer, '');
});
