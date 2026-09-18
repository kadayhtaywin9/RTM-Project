/* Display only: no network calls, location access or SOS payload changes. */
(() => {
  'use strict';
  const clock = document.getElementById('clock');
  const date = document.getElementById('date');
  if (!clock || !date) return;

  const timeZone = 'Asia/Yangon';
  const timeFormat = new Intl.DateTimeFormat('en-GB', {
    timeZone, hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23',
  });
  const dateFormat = new Intl.DateTimeFormat('en-GB', {
    timeZone, weekday: 'short', day: '2-digit', month: 'short', year: 'numeric',
  });
  const isoDateFormat = new Intl.DateTimeFormat('en-GB', {
    timeZone, day: '2-digit', month: '2-digit', year: 'numeric',
  });

  function updateClock() {
    const now = new Date();
    clock.textContent = timeFormat.format(now);
    clock.dateTime = now.toISOString();
    date.textContent = dateFormat.format(now);
    const parts = Object.fromEntries(isoDateFormat.formatToParts(now).map(part => [part.type, part.value]));
    date.dateTime = `${parts.year}-${parts.month}-${parts.day}`;
  }

  updateClock();
  // Read the actual device clock each time; never increment a possibly delayed counter.
  window.setInterval(updateClock, 1000);
  window.addEventListener('pageshow', updateClock);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) updateClock();
  });
})();
