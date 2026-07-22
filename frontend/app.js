const setupSection = document.getElementById('setup-section');
const activeSection = document.getElementById('active-section');
const urlInput = document.getElementById('url-input');
const setupError = document.getElementById('setup-error');

const startBtn = document.getElementById('start-btn');
const modeMenu = document.getElementById('mode-menu');
const stopBtn = document.getElementById('stop-btn');
const switchUnsupervisedBtn = document.getElementById('switch-unsupervised-btn');

const statusMode = document.getElementById('status-mode');
const statusStatus = document.getElementById('status-status');
const statusPages = document.getElementById('status-pages');
const statusFound = document.getElementById('status-found');
const statusQueued = document.getElementById('status-queued');
const statusGraded = document.getElementById('status-graded');
const statusAuto = document.getElementById('status-auto');
const statusCurrentUrl = document.getElementById('status-current-url');
const classCounts = document.getElementById('class-counts');

const gradingSection = document.getElementById('grading-section');
const currentImage = document.getElementById('current-image');
const noImageMessage = document.getElementById('no-image-message');
const judgementText = document.getElementById('judgement-text');
const gradingButtons = document.getElementById('grading-buttons');
const unsupervisedNote = document.getElementById('unsupervised-note');

let sessionId = null;
let mode = null;
let running = false;
let statusTimer = null;
let nextImageAbort = null;
let currentImageId = null;

startBtn.addEventListener('click', () => {
  modeMenu.classList.toggle('hidden');
});

modeMenu.querySelectorAll('button[data-mode]').forEach((btn) => {
  btn.addEventListener('click', () => startCrawl(btn.dataset.mode));
});

stopBtn.addEventListener('click', stopCrawl);
switchUnsupervisedBtn.addEventListener('click', () => setMode('unsupervised'));

gradingButtons.querySelectorAll('button[data-label]').forEach((btn) => {
  btn.addEventListener('click', () => gradeCurrentImage(btn.dataset.label));
});

function parseUrls() {
  return urlInput.value
    .split('\n')
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

async function startCrawl(chosenMode) {
  const seedUrls = parseUrls();
  if (seedUrls.length === 0) {
    setupError.textContent = 'Enter at least one URL to crawl.';
    setupError.classList.remove('hidden');
    return;
  }
  setupError.classList.add('hidden');
  modeMenu.classList.add('hidden');

  const resp = await fetch('/api/crawl/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seed_urls: seedUrls, mode: chosenMode }),
  });
  if (!resp.ok) {
    setupError.textContent = 'Failed to start crawl.';
    setupError.classList.remove('hidden');
    return;
  }
  const data = await resp.json();
  sessionId = data.session_id;
  mode = chosenMode;
  running = true;

  setupSection.classList.add('hidden');
  activeSection.classList.remove('hidden');
  applyModeToUI();

  startStatusPolling();
  if (mode === 'supervised') {
    pollNextImage();
  }
}

async function stopCrawl() {
  running = false;
  if (nextImageAbort) nextImageAbort.abort();
  if (statusTimer) clearInterval(statusTimer);

  if (sessionId) {
    await fetch('/api/crawl/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    });
  }

  sessionId = null;
  mode = null;
  currentImageId = null;
  activeSection.classList.add('hidden');
  setupSection.classList.remove('hidden');
}

async function setMode(newMode) {
  if (!sessionId) return;
  await fetch('/api/crawl/mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, mode: newMode }),
  });
  mode = newMode;
  applyModeToUI();
  // any in-flight next-image long-poll for the old mode is harmless to let finish;
  // the loop checks `mode` before continuing so it will stop naturally.
}

function applyModeToUI() {
  if (mode === 'supervised') {
    gradingSection.classList.remove('hidden');
    unsupervisedNote.classList.add('hidden');
    switchUnsupervisedBtn.classList.remove('hidden');
  } else {
    gradingSection.classList.add('hidden');
    unsupervisedNote.classList.remove('hidden');
    switchUnsupervisedBtn.classList.add('hidden');
  }
}

function startStatusPolling() {
  statusTimer = setInterval(refreshStatus, 1500);
  refreshStatus();
}

async function refreshStatus() {
  if (!sessionId) return;
  const resp = await fetch(`/api/crawl/status?session_id=${encodeURIComponent(sessionId)}`);
  if (!resp.ok) return;
  const s = await resp.json();
  statusMode.textContent = s.mode;
  statusStatus.textContent = s.status;
  statusPages.textContent = s.pages_visited;
  statusFound.textContent = s.images_found;
  statusQueued.textContent = s.images_queued;
  statusGraded.textContent = s.images_graded;
  statusAuto.textContent = s.images_auto_filed;
  statusCurrentUrl.textContent = s.current_url || '';
  classCounts.innerHTML = Object.entries(s.class_counts)
    .map(([label, count]) => `<span class="class-count">${label}: ${count}</span>`)
    .join(' ');
}

async function pollNextImage() {
  if (!running || mode !== 'supervised' || !sessionId) return;

  currentImageId = null;
  currentImage.classList.add('hidden');
  noImageMessage.classList.remove('hidden');
  judgementText.textContent = 'Waiting for image…';
  setGradingButtonsEnabled(false);

  nextImageAbort = new AbortController();
  let resp;
  try {
    resp = await fetch(
      `/api/next-image?session_id=${encodeURIComponent(sessionId)}&timeout=10`,
      { signal: nextImageAbort.signal }
    );
  } catch (err) {
    if (!running) return;
    setTimeout(pollNextImage, 1000);
    return;
  }

  if (!running || mode !== 'supervised') return;

  if (resp.status === 204) {
    pollNextImage();
    return;
  }
  if (!resp.ok) {
    setTimeout(pollNextImage, 1000);
    return;
  }

  const data = await resp.json();
  currentImageId = data.image_id;
  currentImage.src = `/api/image/${data.image_id}?session_id=${encodeURIComponent(sessionId)}`;
  currentImage.classList.remove('hidden');
  noImageMessage.classList.add('hidden');
  judgementText.textContent = describePrediction(data.prediction);
  setGradingButtonsEnabled(true);
}

function describePrediction(prediction) {
  if (!prediction) {
    return 'Not enough labels yet to make a prediction — keep grading!';
  }
  const pct = Math.round(prediction.probs[prediction.label] * 100);
  const labelText = {
    not_part: 'not part of the class',
    part: 'part of the class',
    textbook: 'a textbook example of the class',
  }[prediction.label] || prediction.label;
  return `The model thinks this is ${labelText} (${pct}% confidence).`;
}

function setGradingButtonsEnabled(enabled) {
  gradingButtons.querySelectorAll('button').forEach((btn) => {
    btn.disabled = !enabled;
  });
}

async function gradeCurrentImage(label) {
  if (!currentImageId || !sessionId) return;
  setGradingButtonsEnabled(false);
  await fetch('/api/grade', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, image_id: currentImageId, label }),
  });
  pollNextImage();
}
