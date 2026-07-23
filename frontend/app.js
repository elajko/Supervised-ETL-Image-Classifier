const modelSelect = document.getElementById('model-select');
const renameModelBtn = document.getElementById('rename-model-btn');
const newModelBtn = document.getElementById('new-model-btn');
const newModelForm = document.getElementById('new-model-form');
const newModelNameInput = document.getElementById('new-model-name');
const createModelBtn = document.getElementById('create-model-btn');
const cancelNewModelBtn = document.getElementById('cancel-new-model-btn');
const renameModelForm = document.getElementById('rename-model-form');
const renameModelNameInput = document.getElementById('rename-model-name');
const confirmRenameBtn = document.getElementById('confirm-rename-btn');
const cancelRenameBtn = document.getElementById('cancel-rename-btn');
const modelError = document.getElementById('model-error');

const setupSection = document.getElementById('setup-section');
const activeSection = document.getElementById('active-section');
const urlInput = document.getElementById('url-input');
const setupError = document.getElementById('setup-error');

const startBtn = document.getElementById('start-btn');
const modeMenu = document.getElementById('mode-menu');
const stopBtn = document.getElementById('stop-btn');
const switchUnsupervisedBtn = document.getElementById('switch-unsupervised-btn');
const testClassifierBtn = document.getElementById('test-classifier-btn');
const testClassifierStatus = document.getElementById('test-classifier-status');
const reviewAutoBtn = document.getElementById('review-auto-btn');

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

const gallerySection = document.getElementById('gallery-section');
const galleryCount = document.getElementById('gallery-count');
const galleryGrid = document.getElementById('gallery-grid');
const galleryFilterClassification = document.getElementById('gallery-filter-classification');
const galleryFilterSource = document.getElementById('gallery-filter-source');

galleryFilterClassification.addEventListener('change', refreshGallery);
galleryFilterSource.addEventListener('change', refreshGallery);

const STORAGE_KEY = 'etl-classifier-current-model-id';

let modelsById = {};
let currentModelId = null;
let mode = null;
let running = false;
let statusTimer = null;
let nextImageAbort = null;
let currentImageId = null;

let testingActive = false;
let testingResults = [];
let testingIndex = 0;
let testingTimer = null;

let reviewingAuto = false;
let currentAutoImageId = null;

init();

async function init() {
  await refreshModelList();
  const savedId = localStorage.getItem(STORAGE_KEY);
  if (savedId && modelsById[savedId]) {
    modelSelect.value = savedId;
    await selectModel(savedId);
  }
}

modelSelect.addEventListener('change', () => {
  const id = modelSelect.value;
  if (id) selectModel(id);
});

newModelBtn.addEventListener('click', () => {
  newModelForm.classList.remove('hidden');
  renameModelForm.classList.add('hidden');
  newModelNameInput.value = '';
  newModelNameInput.focus();
});

cancelNewModelBtn.addEventListener('click', () => {
  newModelForm.classList.add('hidden');
});

createModelBtn.addEventListener('click', async () => {
  const name = newModelNameInput.value.trim();
  if (!name) {
    showModelError('Enter a name for the new model.');
    return;
  }
  const resp = await fetch('/api/models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  if (!resp.ok) {
    showModelError('Failed to create model.');
    return;
  }
  hideModelError();
  newModelForm.classList.add('hidden');
  await refreshModelList();
  modelSelect.value = (await resp.json()).session_id;
  await selectModel(modelSelect.value);
});

renameModelBtn.addEventListener('click', () => {
  if (!currentModelId) return;
  renameModelNameInput.value = modelsById[currentModelId].name;
  renameModelForm.classList.remove('hidden');
  newModelForm.classList.add('hidden');
  renameModelNameInput.focus();
});

cancelRenameBtn.addEventListener('click', () => {
  renameModelForm.classList.add('hidden');
});

confirmRenameBtn.addEventListener('click', async () => {
  const name = renameModelNameInput.value.trim();
  if (!name || !currentModelId) {
    showModelError('Enter a name.');
    return;
  }
  const resp = await fetch(`/api/models/${encodeURIComponent(currentModelId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  if (!resp.ok) {
    showModelError('Failed to rename model.');
    return;
  }
  hideModelError();
  renameModelForm.classList.add('hidden');
  await refreshModelList();
  modelSelect.value = currentModelId;
});

function showModelError(text) {
  modelError.textContent = text;
  modelError.classList.remove('hidden');
}

function hideModelError() {
  modelError.classList.add('hidden');
}

async function refreshModelList() {
  const resp = await fetch('/api/models');
  if (!resp.ok) return;
  const models = await resp.json();
  modelsById = Object.fromEntries(models.map((m) => [m.session_id, m]));

  const previousValue = modelSelect.value;
  modelSelect.innerHTML = '<option value="">Select a model&hellip;</option>';
  for (const m of models) {
    const option = document.createElement('option');
    option.value = m.session_id;
    option.textContent = formatModelLabel(m);
    modelSelect.appendChild(option);
  }
  if (previousValue && modelsById[previousValue]) {
    modelSelect.value = previousValue;
  }
}

function formatModelLabel(model) {
  const good = model.class_counts.good || 0;
  const great = model.class_counts.great || 0;
  return `${model.name} (good: ${good}, great: ${great})`;
}

async function selectModel(id) {
  stopWatching();
  stopTesting();
  stopReviewing();
  currentModelId = id;
  localStorage.setItem(STORAGE_KEY, id);
  renameModelBtn.classList.remove('hidden');
  gallerySection.classList.remove('hidden');
  currentImageId = null;

  await refreshGallery();

  const resp = await fetch(`/api/crawl/status?session_id=${encodeURIComponent(id)}`);
  if (!resp.ok) return;
  const status = await resp.json();
  mode = status.mode;

  if (status.status === 'crawling') {
    setupSection.classList.add('hidden');
    activeSection.classList.remove('hidden');
    running = true;
    applyModeToUI();
    startStatusPolling();
    if (mode === 'supervised') pollNextImage();
  } else {
    activeSection.classList.add('hidden');
    setupSection.classList.remove('hidden');
  }
}

function parseUrls() {
  return urlInput.value
    .split('\n')
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

startBtn.addEventListener('click', () => {
  modeMenu.classList.toggle('hidden');
});

modeMenu.querySelectorAll('button[data-mode]').forEach((btn) => {
  btn.addEventListener('click', () => startCrawl(btn.dataset.mode));
});

stopBtn.addEventListener('click', stopCrawl);
switchUnsupervisedBtn.addEventListener('click', () => setMode('unsupervised'));
testClassifierBtn.addEventListener('click', () => {
  if (testingActive) {
    stopTesting();
  } else {
    startTesting();
  }
});
reviewAutoBtn.addEventListener('click', () => {
  if (reviewingAuto) {
    stopReviewing();
  } else {
    startReviewing();
  }
});

gradingButtons.querySelectorAll('button[data-label]').forEach((btn) => {
  btn.addEventListener('click', () => {
    if (reviewingAuto) {
      promoteCurrentAutoImage(btn.dataset.label);
    } else {
      gradeCurrentImage(btn.dataset.label);
    }
  });
});

async function startCrawl(chosenMode) {
  if (!currentModelId) return;
  const seedUrls = parseUrls();
  if (seedUrls.length === 0) {
    setupError.textContent = 'Enter at least one URL to crawl.';
    setupError.classList.remove('hidden');
    return;
  }
  setupError.classList.add('hidden');
  modeMenu.classList.add('hidden');
  stopReviewing();

  const resp = await fetch('/api/crawl/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: currentModelId, seed_urls: seedUrls, mode: chosenMode }),
  });
  if (!resp.ok) {
    setupError.textContent = 'Failed to start crawl.';
    setupError.classList.remove('hidden');
    return;
  }
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

// Stops the frontend's local polling loops without touching the backend
// crawl — used when switching to a different model, since the previous
// model's crawl should keep running server-side regardless of whether
// anyone's watching it.
function stopWatching() {
  running = false;
  if (nextImageAbort) nextImageAbort.abort();
  if (statusTimer) clearInterval(statusTimer);
}

async function stopCrawl() {
  if (!currentModelId) return;
  stopWatching();
  await fetch('/api/crawl/stop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: currentModelId }),
  });
  currentImageId = null;
  activeSection.classList.add('hidden');
  setupSection.classList.remove('hidden');
  await refreshModelList();
  modelSelect.value = currentModelId;
}

async function setMode(newMode) {
  if (!currentModelId) return;
  await fetch('/api/crawl/mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: currentModelId, mode: newMode }),
  });
  mode = newMode;
  applyModeToUI();
  // any in-flight next-image long-poll for the old mode is harmless to let finish;
  // the loop checks `mode` before continuing so it will stop naturally.
}

// grading-section is shared by two mutually-exclusive activities: grading
// images live during a supervised crawl, and reviewing already-saved
// auto-filed images. Show it whenever either is active.
function updateGradingSectionVisibility() {
  const showForCrawl = running && mode === 'supervised';
  if (showForCrawl || reviewingAuto) {
    gradingSection.classList.remove('hidden');
  } else {
    gradingSection.classList.add('hidden');
  }
}

function applyModeToUI() {
  updateGradingSectionVisibility();
  if (mode === 'supervised') {
    unsupervisedNote.classList.add('hidden');
    switchUnsupervisedBtn.classList.remove('hidden');
  } else {
    unsupervisedNote.classList.remove('hidden');
    switchUnsupervisedBtn.classList.add('hidden');
  }
}

async function startTesting() {
  if (!currentModelId) return;
  stopReviewing();
  const modelId = currentModelId;
  const resp = await fetch(`/api/models/${encodeURIComponent(modelId)}/test-classifier`);
  if (!resp.ok || currentModelId !== modelId) return;
  const results = await resp.json();
  if (currentModelId !== modelId) return;

  testingResults = results;
  testingIndex = 0;
  testingActive = true;
  testClassifierBtn.textContent = 'End testing early';
  testClassifierStatus.classList.remove('hidden');
  advanceTesting();
}

function stopTesting() {
  testingActive = false;
  if (testingTimer) clearTimeout(testingTimer);
  testClassifierBtn.textContent = 'Test classifier on manually classified images';
  testClassifierStatus.classList.add('hidden');
}

async function startReviewing() {
  if (!currentModelId) return;
  stopWatching();
  stopTesting();
  reviewingAuto = true;
  reviewAutoBtn.textContent = 'Stop reviewing';
  updateGradingSectionVisibility();
  pollNextAutoImage();
}

function stopReviewing() {
  reviewingAuto = false;
  currentAutoImageId = null;
  reviewAutoBtn.textContent = 'Supervise result of unsupervised images';
  updateGradingSectionVisibility();
}

async function pollNextAutoImage() {
  if (!reviewingAuto || !currentModelId) return;
  const modelId = currentModelId;

  currentAutoImageId = null;
  currentImage.classList.add('hidden');
  noImageMessage.textContent = 'Loading next image to review…';
  noImageMessage.classList.remove('hidden');
  setJudgementNeutral('Loading…');
  setGradingButtonsEnabled(false);

  const resp = await fetch(`/api/models/${encodeURIComponent(modelId)}/next-auto-image`);
  if (!reviewingAuto || currentModelId !== modelId) return;

  if (resp.status === 204) {
    noImageMessage.textContent = 'No more auto-classified images to review.';
    setJudgementNeutral('Nothing left to review.');
    stopReviewing();
    return;
  }
  if (!resp.ok) {
    setTimeout(pollNextAutoImage, 1000);
    return;
  }

  const data = await resp.json();
  currentAutoImageId = data.image_id;
  currentImage.src = `/api/image/${data.image_id}?session_id=${encodeURIComponent(modelId)}`;
  currentImage.classList.remove('hidden');
  noImageMessage.classList.add('hidden');
  renderAutoClassification(data.classification);
  setGradingButtonsEnabled(true);
}

function renderAutoClassification(classification) {
  const isPositive = classification !== 'not_part';
  const labelText = {
    not_part: 'not part of the class',
    good: 'part of the class',
    great: 'a textbook example of the class',
  }[classification] || classification;

  const icon = document.createElement('span');
  icon.className = 'judgement-icon';
  icon.setAttribute('aria-hidden', 'true');
  icon.textContent = isPositive ? '✓' : '✗';

  const message = document.createElement('span');
  message.textContent = `This was auto-classified as ${labelText}. Confirm or correct below.`;

  judgementText.className = isPositive ? 'judgement-positive' : 'judgement-negative';
  judgementText.replaceChildren(icon, message);
}

async function promoteCurrentAutoImage(label) {
  if (!currentAutoImageId || !currentModelId) return;
  setGradingButtonsEnabled(false);
  await fetch(`/api/models/${encodeURIComponent(currentModelId)}/promote-image`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: currentAutoImageId, label }),
  });
  await refreshGallery();
  await refreshModelList();
  if (currentModelId) modelSelect.value = currentModelId;
  pollNextAutoImage();
}

function advanceTesting() {
  if (!testingActive) return;
  if (testingResults.length === 0 || testingIndex >= testingResults.length) {
    stopTesting();
    return;
  }
  const seenSoFar = testingResults.slice(0, testingIndex + 1);
  const correctSoFar = seenSoFar.filter((r) => r.correct).length;
  const pct = Math.round((correctSoFar / seenSoFar.length) * 100);
  const left = testingResults.length - (testingIndex + 1);
  testClassifierStatus.textContent = `Images left to test: ${left}; Current correct classification rate: ${pct}%`;
  testingIndex++;
  testingTimer = setTimeout(advanceTesting, 80);
}

function startStatusPolling() {
  statusTimer = setInterval(refreshStatus, 1500);
  refreshStatus();
}

async function refreshStatus() {
  if (!currentModelId) return;
  const resp = await fetch(`/api/crawl/status?session_id=${encodeURIComponent(currentModelId)}`);
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
  if (!running || mode !== 'supervised' || !currentModelId) return;

  currentImageId = null;
  currentImage.classList.add('hidden');
  noImageMessage.classList.remove('hidden');
  setJudgementNeutral('Waiting for image…');
  setGradingButtonsEnabled(false);

  const modelId = currentModelId;
  nextImageAbort = new AbortController();
  let resp;
  try {
    resp = await fetch(
      `/api/next-image?session_id=${encodeURIComponent(modelId)}&timeout=10`,
      { signal: nextImageAbort.signal }
    );
  } catch (err) {
    if (!running || currentModelId !== modelId) return;
    setTimeout(pollNextImage, 1000);
    return;
  }

  if (!running || mode !== 'supervised' || currentModelId !== modelId) return;

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
  currentImage.src = `/api/image/${data.image_id}?session_id=${encodeURIComponent(modelId)}`;
  currentImage.classList.remove('hidden');
  noImageMessage.classList.add('hidden');
  renderPrediction(data.prediction);
  setGradingButtonsEnabled(true);
}

function setJudgementNeutral(text) {
  judgementText.className = '';
  judgementText.textContent = text;
}

function renderPrediction(prediction) {
  if (!prediction) {
    setJudgementNeutral('Not enough labels yet to make a prediction — keep grading!');
    return;
  }

  const isPositive = prediction.label !== 'not_part';
  const pct = Math.round(prediction.probs[prediction.label] * 100);
  const labelText = {
    not_part: 'not part of the class',
    good: 'part of the class',
    great: 'a textbook example of the class',
  }[prediction.label] || prediction.label;

  const icon = document.createElement('span');
  icon.className = 'judgement-icon';
  icon.setAttribute('aria-hidden', 'true');
  icon.textContent = isPositive ? '✓' : '✗';

  const pctEl = document.createElement('strong');
  pctEl.textContent = `${pct}%`;

  const message = document.createElement('span');
  message.append(
    `The model thinks this is ${labelText} (`,
    pctEl,
    ' confidence).'
  );

  judgementText.className = isPositive ? 'judgement-positive' : 'judgement-negative';
  judgementText.replaceChildren(icon, message);
}

function setGradingButtonsEnabled(enabled) {
  gradingButtons.querySelectorAll('button').forEach((btn) => {
    btn.disabled = !enabled;
  });
}

async function gradeCurrentImage(label) {
  if (!currentImageId || !currentModelId) return;
  setGradingButtonsEnabled(false);
  await fetch('/api/grade', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: currentModelId, image_id: currentImageId, label }),
  });
  await refreshGallery();
  await refreshModelList();
  if (currentModelId) modelSelect.value = currentModelId;
  pollNextImage();
}

function galleryBadgeClass(label) {
  if (label === 'good' || label === 'auto-good') return 'good';
  if (label === 'great' || label === 'auto-great') return 'great';
  if (label === 'not_part' || label === 'auto_not_part') return 'not-part';
  return '';
}

async function refreshGallery() {
  if (!currentModelId) return;
  const modelId = currentModelId;
  const params = new URLSearchParams();
  if (galleryFilterClassification.value) params.set('classification', galleryFilterClassification.value);
  if (galleryFilterSource.value) params.set('source', galleryFilterSource.value);
  const resp = await fetch(`/api/models/${encodeURIComponent(modelId)}/images?${params}`);
  if (!resp.ok || currentModelId !== modelId) return;
  const images = await resp.json();
  if (currentModelId !== modelId) return;

  galleryCount.textContent = images.length;
  galleryGrid.replaceChildren();
  for (const img of images) {
    const cell = document.createElement('div');
    cell.className = 'gallery-cell';

    const thumb = document.createElement('img');
    thumb.src = `/api/image/${img.image_id}?session_id=${encodeURIComponent(modelId)}`;
    thumb.alt = img.label;
    thumb.loading = 'lazy';

    const badge = document.createElement('span');
    const badgeClass = galleryBadgeClass(img.label);
    badge.className = badgeClass ? `gallery-badge gallery-badge-${badgeClass}` : 'gallery-badge';
    badge.textContent = img.label;

    cell.append(thumb, badge);
    galleryGrid.appendChild(cell);
  }
}
