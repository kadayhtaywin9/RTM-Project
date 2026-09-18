// Deterministic tests of the production clock; no browser, GPS or network needed.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../sos_service/static/sos-clock.js', import.meta.url), 'utf8');
const html = fs.readFileSync(new URL('../sos_service/static/index.html', import.meta.url), 'utf8');
assert.ok(html.indexOf('/static/sos-clock.js?v=19') < html.indexOf('leaflet@1.9.4/dist/leaflet.js'));
assert.ok(html.includes('role="timer"') && html.includes('aria-live="off"'));

for (const deviceZone of ['UTC', 'America/New_York', 'Asia/Tokyo']) {
  process.env.TZ = deviceZone;
  let instant = '2026-09-14T17:29:59Z';
  const nodes = {clock: {}, date: {}};
  const events = {};
  const timers = [];
  class TestDate extends Date { constructor(...args) { super(...(args.length ? args : [instant])); } }
  const document = {hidden: false, getElementById: id => nodes[id],
    addEventListener: (name, handler) => { events[name] = handler; }};
  vm.runInNewContext(source, {
    Date: TestDate, Intl, document,
    window: {setInterval: (handler, delay) => timers.push({handler, delay}),
      addEventListener: (name, handler) => { events[name] = handler; }},
  });
  assert.equal(nodes.clock.textContent, '23:59:59', 'render immediately without waiting for a timer');
  assert.equal(nodes.date.dateTime, '2026-09-14');
  assert.equal(timers.length, 1);
  assert.equal(timers[0].delay, 1000);
  instant = '2026-09-14T17:30:00Z';
  timers[0].handler();
  assert.equal(nodes.clock.textContent, '00:00:00');
  assert.equal(nodes.date.dateTime, '2026-09-15', 'date must roll over in Myanmar, not UTC');
  assert.ok(nodes.date.textContent.includes('15 Sept 2026'));
  document.hidden = true;
  instant = '2026-12-31T17:30:00Z';
  events.visibilitychange();
  assert.equal(nodes.date.dateTime, '2026-09-15');
  document.hidden = false;
  events.visibilitychange();
  assert.equal(nodes.date.dateTime, '2027-01-01');
  assert.equal(nodes.clock.textContent, '00:00:00', 'resume reads current time rather than adding missed ticks');
  instant = '2027-01-01T04:05:06Z';
  events.pageshow();
  assert.equal(nodes.clock.textContent, '10:35:06');
  assert.equal(nodes.clock.dateTime, '2027-01-01T04:05:06.000Z');
}
console.log('SOS clock passed: immediate rendering, seconds, Myanmar midnight/year rollover, device timezone independence and resume.');
