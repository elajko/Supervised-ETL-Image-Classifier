const modelSelect = document.getElementById('model-select');
const renameModelBtn = document.getElementById('rename-model-btn');
const deleteModelBtn = document.getElementById('delete-model-btn');
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
const crawlModeSelect = document.getElementById('crawl-mode-select');
const stopBtn = document.getElementById('stop-btn');
const switchUnsupervisedBtn = document.getElementById('switch-unsupervised-btn');
const testRegressorBtn = document.getElementById('test-regressor-btn');
const testRegressorStatus = document.getElementById('test-regressor-status');
const reviewAutoBtn = document.getElementById('review-auto-btn');
const sourceSetupPanel = document.getElementById('source-setup-panel');
const sourceSetupList = document.getElementById('source-setup-list');
const saveThresholdSlider = document.getElementById('save-threshold-slider');
const saveThresholdValue = document.getElementById('save-threshold-value');

const statusMode = document.getElementById('status-mode');
const statusStatus = document.getElementById('status-status');
const statusPages = document.getElementById('status-pages');
const statusFound = document.getElementById('status-found');
const statusQueued = document.getElementById('status-queued');
const statusGraded = document.getElementById('status-graded');
const statusAuto = document.getElementById('status-auto');
const statusCurrentUrl = document.getElementById('status-current-url');
const statusLastError = document.getElementById('status-last-error');

const gradingSection = document.getElementById('grading-section');
const rightPanePlaceholder = document.getElementById('right-pane-placeholder');
const currentImage = document.getElementById('current-image');
const noImageMessage = document.getElementById('no-image-message');
const judgementText = document.getElementById('judgement-text');
const pageInfoRow = document.getElementById('page-info-row');
const pageInfoLink = document.getElementById('page-info-link');
const skipPageBtn = document.getElementById('skip-page-btn');
const scoreSlider = document.getElementById('score-slider');
const scoreSliderValue = document.getElementById('score-slider-value');
const submitScoreBtn = document.getElementById('submit-score-btn');
const deleteImageBtn = document.getElementById('delete-image-btn');
const unsupervisedNote = document.getElementById('unsupervised-note');

const insightsSection = document.getElementById('insights-section');
const galleryCount = document.getElementById('gallery-count');
const galleryGrid = document.getElementById('gallery-grid');
const galleryScoreFilterMin = document.getElementById('gallery-score-filter-min');
const galleryScoreFilterMax = document.getElementById('gallery-score-filter-max');
const galleryScoreFilterFill = document.getElementById('gallery-score-filter-fill');
const galleryScoreFilterValue = document.getElementById('gallery-score-filter-value');
const galleryFilterSource = document.getElementById('gallery-filter-source');
const galleryFilterDomain = document.getElementById('gallery-filter-domain');
const galleryPrevBtn = document.getElementById('gallery-prev-btn');
const galleryNextBtn = document.getElementById('gallery-next-btn');
const galleryPageInfo = document.getElementById('gallery-page-info');
const gallerySortRadios = document.querySelectorAll('#gallery-sort input[name="gallery-sort"]');

const siteStatsCount = document.getElementById('site-stats-count');
const siteStatsList = document.getElementById('site-stats-list');

const scoreHistogramHuman = document.getElementById('score-histogram-human');
const scoreHistogramHumanCount = document.getElementById('score-histogram-human-count');
const scoreHistogramAuto = document.getElementById('score-histogram-auto');
const scoreHistogramAutoCount = document.getElementById('score-histogram-auto-count');
const SCORE_HISTOGRAM_BINS = 100;

document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    });
    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');
    document.getElementById(btn.dataset.panel).classList.add('active');
  });
});

const GALLERY_PAGE_SIZE = 25;
let galleryOffset = 0;

// Dual-thumb range slider: dragging either knob past the other snaps them
// together rather than letting them cross, and the fill bar between them
// mirrors the highlighted portion of a normal single-knob slider.
function updateScoreFilterVisual() {
  const min = Number(galleryScoreFilterMin.value);
  const max = Number(galleryScoreFilterMax.value);
  galleryScoreFilterFill.style.left = `${min}%`;
  galleryScoreFilterFill.style.width = `${max - min}%`;
  galleryScoreFilterValue.textContent = `${min} – ${max}`;
}

function handleScoreFilterInput(dragged) {
  const min = Number(galleryScoreFilterMin.value);
  const max = Number(galleryScoreFilterMax.value);
  if (min > max) {
    if (dragged === 'min') {
      galleryScoreFilterMax.value = min;
    } else {
      galleryScoreFilterMin.value = max;
    }
  }
  updateScoreFilterVisual();
}

galleryScoreFilterMin.addEventListener('input', () => handleScoreFilterInput('min'));
galleryScoreFilterMax.addEventListener('input', () => handleScoreFilterInput('max'));
galleryScoreFilterMin.addEventListener('change', () => {
  galleryOffset = 0;
  refreshGallery();
});
galleryScoreFilterMax.addEventListener('change', () => {
  galleryOffset = 0;
  refreshGallery();
});
updateScoreFilterVisual();

galleryFilterSource.addEventListener('change', () => {
  galleryOffset = 0;
  refreshGallery();
});
galleryFilterDomain.addEventListener('change', () => {
  galleryOffset = 0;
  refreshGallery();
});
gallerySortRadios.forEach((radio) => {
  radio.addEventListener('change', () => {
    galleryOffset = 0;
    refreshGallery();
  });
});
galleryPrevBtn.addEventListener('click', () => {
  galleryOffset = Math.max(0, galleryOffset - GALLERY_PAGE_SIZE);
  refreshGallery();
});
galleryNextBtn.addEventListener('click', () => {
  galleryOffset += GALLERY_PAGE_SIZE;
  refreshGallery();
});

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

let manualReclassifying = false;
let manualImageId = null;

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

// Unlike image deletion, there's no shift-click bypass here on purpose --
// deleting an entire model (every image and its regressor) is much harder
// to shrug off, so the confirmation is never skippable.
deleteModelBtn.addEventListener('click', async () => {
  if (!currentModelId) return;
  const model = modelsById[currentModelId];
  const name = model ? model.name : 'this model';
  const confirmed = confirm(`Delete model "${name}" and all its images? This cannot be undone.`);
  if (!confirmed) return;

  const resp = await fetch(`/api/models/${encodeURIComponent(currentModelId)}`, { method: 'DELETE' });
  if (!resp.ok) {
    showModelError('Failed to delete model.');
    return;
  }
  hideModelError();
  stopWatching();
  stopTesting();
  stopReviewing();
  if (localStorage.getItem(STORAGE_KEY) === currentModelId) {
    localStorage.removeItem(STORAGE_KEY);
  }
  currentModelId = null;
  modelSelect.value = '';
  renameModelBtn.classList.add('hidden');
  deleteModelBtn.classList.add('hidden');
  insightsSection.classList.add('hidden');
  setupSection.classList.add('hidden');
  activeSection.classList.add('hidden');
  await refreshModelList();
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
    option.textContent = m.name;
    modelSelect.appendChild(option);
  }
  if (previousValue && modelsById[previousValue]) {
    modelSelect.value = previousValue;
  }
}

async function selectModel(id) {
  stopWatching();
  stopTesting();
  stopReviewing();
  currentModelId = id;
  localStorage.setItem(STORAGE_KEY, id);
  renameModelBtn.classList.remove('hidden');
  deleteModelBtn.classList.remove('hidden');
  insightsSection.classList.remove('hidden');
  currentImageId = null;
  galleryOffset = 0;

  await refreshGallery();
  await refreshSiteStats();
  await refreshScoreHistogram();

  const resp = await fetch(`/api/crawl/status?session_id=${encodeURIComponent(id)}`);
  if (!resp.ok) return;
  const status = await resp.json();
  mode = status.mode;
  setSaveThresholdValue(status.save_threshold);

  if (status.status === 'crawling') {
    setupSection.classList.add('hidden');
    activeSection.classList.remove('hidden');
    running = true;
    updateGalleryLockState();
    applyModeToUI();
    startStatusPolling();
    if (mode === 'supervised') pollNextImage();
  } else {
    activeSection.classList.add('hidden');
    setupSection.classList.remove('hidden');
    updateGalleryLockState();
  }
}

function parseUrls() {
  return urlInput.value
    .split('\n')
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

startBtn.addEventListener('click', async () => {
  const seedUrls = parseUrls();
  if (seedUrls.length === 0) {
    setupError.textContent = 'Enter at least one URL to crawl.';
    setupError.classList.remove('hidden');
    return;
  }
  setupError.classList.add('hidden');
  const ready = await checkSourceReadiness(seedUrls);
  if (ready) {
    await startCrawl(crawlModeSelect.value);
  }
});

// Detects source-adapter-handled domains (Imgur/DeviantArt/Pinterest) among
// the seed URLs and, if any need credentials or interactive login that
// aren't set up yet, blocks the crawl behind a setup panel instead.
function hostnameOf(url) {
  try {
    return new URL(url).hostname.toLowerCase();
  } catch {
    return null;
  }
}

function matchSource(url, sources) {
  const host = hostnameOf(url);
  if (!host) return null;
  const bare = host.startsWith('www.') ? host.slice(4) : host;
  return sources.find((s) => s.domains.some((d) => bare === d || bare.endsWith('.' + d))) || null;
}

// Returns true once every source used by the seed URLs is ready to crawl.
async function checkSourceReadiness(seedUrls) {
  const resp = await fetch('/api/sources');
  const sources = resp.ok ? await resp.json() : [];
  const matchedNames = new Set();
  for (const url of seedUrls) {
    const m = matchSource(url, sources);
    if (m) matchedNames.add(m.name);
  }
  const matched = sources.filter((s) => matchedNames.has(s.name));
  const pending = matched.filter((s) => !s.configured || !s.authenticated);

  if (pending.length === 0) {
    sourceSetupPanel.classList.add('hidden');
    return true;
  }

  sourceSetupPanel.classList.remove('hidden');
  sourceSetupList.replaceChildren();
  for (const source of pending) {
    sourceSetupList.appendChild(buildSourceSetupCard(source, seedUrls));
  }
  return false;
}

function buildSourceSetupCard(source, seedUrls) {
  const card = document.createElement('div');
  card.className = 'source-setup-card';

  const title = document.createElement('strong');
  title.textContent = source.name;
  card.appendChild(title);

  const cardError = document.createElement('p');
  cardError.className = 'error hidden';
  card.appendChild(cardError);

  if (!source.configured) {
    const idInput = document.createElement('input');
    idInput.type = 'text';
    idInput.placeholder = 'Client ID';

    let secretInput = null;
    if (source.needs_client_secret) {
      secretInput = document.createElement('input');
      secretInput.type = 'password';
      secretInput.placeholder = 'Client Secret';
    }

    const saveBtn = document.createElement('button');
    saveBtn.className = 'primary';
    saveBtn.textContent = 'Save';
    saveBtn.addEventListener('click', async () => {
      const clientId = idInput.value.trim();
      if (!clientId) {
        cardError.textContent = 'Enter a Client ID.';
        cardError.classList.remove('hidden');
        return;
      }
      const body = { client_id: clientId };
      if (secretInput) body.client_secret = secretInput.value.trim();
      const resp = await fetch(`/api/sources/${encodeURIComponent(source.name)}/credentials`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        cardError.textContent = err.detail || 'Failed to save credentials.';
        cardError.classList.remove('hidden');
        return;
      }
      await checkSourceReadiness(seedUrls);
    });

    card.append(idInput);
    if (secretInput) card.append(secretInput);
    card.append(saveBtn);
  } else if (source.supports_interactive_auth) {
    const authBtn = document.createElement('button');
    authBtn.className = 'primary';
    authBtn.textContent = `Authenticate with ${source.name}`;
    authBtn.addEventListener('click', async () => {
      const resp = await fetch(`/api/sources/${encodeURIComponent(source.name)}/auth-url`);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        cardError.textContent = err.detail || 'Failed to start authentication.';
        cardError.classList.remove('hidden');
        return;
      }
      const { url } = await resp.json();
      window.open(url, '_blank', 'width=600,height=700');
      pollSourceAuthStatus(source.name, seedUrls);
    });
    card.append(authBtn);
  } else {
    const msg = document.createElement('span');
    msg.textContent = 'Not available right now.';
    card.append(msg);
  }

  return card;
}

function pollSourceAuthStatus(siteName, seedUrls) {
  const timer = setInterval(async () => {
    if (sourceSetupPanel.classList.contains('hidden')) {
      clearInterval(timer);
      return;
    }
    const resp = await fetch(`/api/sources/${encodeURIComponent(siteName)}/status`);
    if (!resp.ok) return;
    const status = await resp.json();
    if (status.authenticated) {
      clearInterval(timer);
      await checkSourceReadiness(seedUrls);
    }
  }, 1500);
}

stopBtn.addEventListener('click', stopCrawl);
switchUnsupervisedBtn.addEventListener('click', () => setMode('unsupervised'));
testRegressorBtn.addEventListener('click', () => {
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

scoreSlider.addEventListener('input', () => {
  scoreSliderValue.textContent = scoreSlider.value;
});

submitScoreBtn.addEventListener('click', () => {
  const score = Number(scoreSlider.value);
  if (manualReclassifying) {
    submitManualReclassify(score);
  } else if (reviewingAuto) {
    promoteCurrentAutoImage(score);
  } else {
    gradeCurrentImage(score);
  }
});

saveThresholdSlider.addEventListener('input', () => {
  saveThresholdValue.textContent = saveThresholdSlider.value;
});

function setSaveThresholdValue(value) {
  const rounded = Math.round(value);
  saveThresholdSlider.value = rounded;
  saveThresholdValue.textContent = rounded;
}

deleteImageBtn.addEventListener('click', (e) => {
  if (!manualReclassifying || !manualImageId) return;
  if (!e.shiftKey) {
    const confirmed = confirm(
      'Confirm deleting this image? You can skip this pop-up by holding shift when pressing the delete button.'
    );
    if (!confirmed) return;
  }
  deleteCurrentManualImage();
});

async function deleteCurrentManualImage() {
  if (!manualImageId || !currentModelId) return;
  const imageId = manualImageId;
  setScoreControlsEnabled(false);
  await fetch(`/api/models/${encodeURIComponent(currentModelId)}/images/${encodeURIComponent(imageId)}`, {
    method: 'DELETE',
  });
  manualReclassifying = false;
  manualImageId = null;
  updateGradingSectionVisibility();
  galleryOffset = 0;
  await refreshGallery();
  await refreshSiteStats();
  await refreshScoreHistogram();
  await refreshModelList();
  if (currentModelId) modelSelect.value = currentModelId;
}

async function startCrawl(chosenMode) {
  if (!currentModelId) return;
  const seedUrls = parseUrls();
  if (seedUrls.length === 0) {
    setupError.textContent = 'Enter at least one URL to crawl.';
    setupError.classList.remove('hidden');
    return;
  }
  setupError.classList.add('hidden');
  stopReviewing();

  const resp = await fetch('/api/crawl/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: currentModelId,
      seed_urls: seedUrls,
      mode: chosenMode,
      save_threshold: Number(saveThresholdSlider.value),
    }),
  });
  if (!resp.ok) {
    setupError.textContent = 'Failed to start crawl.';
    setupError.classList.remove('hidden');
    return;
  }
  mode = chosenMode;
  running = true;
  updateGalleryLockState();

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
  updateGalleryLockState();
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

// grading-section is shared by three mutually-exclusive activities: grading
// images live during a supervised crawl, reviewing already-saved auto-filed
// images, and manually reclassifying a gallery image. Show it whenever any
// of those is active.
function updateGradingSectionVisibility() {
  const showForCrawl = running && mode === 'supervised';
  const active = showForCrawl || reviewingAuto || manualReclassifying;
  gradingSection.classList.toggle('hidden', !active);
  rightPanePlaceholder.classList.toggle('hidden', active);
  // Deleting only makes sense for an already-saved image being manually
  // reclassified from the gallery -- not a live-crawl image still pending
  // its first grade, nor an auto-filed image mid-review.
  deleteImageBtn.classList.toggle('hidden', !manualReclassifying);
  // "Skip this page" only means something while a crawl is actively
  // pulling images -- there's nothing left to affect once it's stopped.
  if (!showForCrawl) pageInfoRow.classList.add('hidden');
}

// Reclassifying a gallery image from the gallery itself is only safe while
// the crawler isn't actively running, to avoid racing with the live-crawl
// grading flow that also drives the same right-pane controls. The save
// threshold is likewise locked while running since it's read once at crawl
// start and only takes effect for a fresh run.
function updateGalleryLockState() {
  galleryGrid.classList.toggle('locked', running);
  saveThresholdSlider.disabled = running;
}

function applyModeToUI() {
  updateGradingSectionVisibility();
  crawlModeSelect.value = mode;
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
  const resp = await fetch(`/api/models/${encodeURIComponent(modelId)}/test-regressor`);
  if (!resp.ok || currentModelId !== modelId) return;
  const results = await resp.json();
  if (currentModelId !== modelId) return;

  testingResults = results;
  testingIndex = 0;
  testingActive = true;
  testRegressorBtn.textContent = 'End testing early';
  testRegressorStatus.classList.remove('hidden');
  advanceTesting();
}

function stopTesting() {
  testingActive = false;
  if (testingTimer) clearTimeout(testingTimer);
  testRegressorBtn.textContent = 'Test classifier on manually classified images';
  testRegressorStatus.classList.add('hidden');
}

function advanceTesting() {
  if (!testingActive) return;
  if (testingResults.length === 0 || testingIndex >= testingResults.length) {
    stopTesting();
    return;
  }
  const seenSoFar = testingResults.slice(0, testingIndex + 1);
  const validErrors = seenSoFar.filter((r) => r.error !== null).map((r) => r.error);
  const avgError = validErrors.length ? validErrors.reduce((a, b) => a + b, 0) / validErrors.length : 0;
  const left = testingResults.length - (testingIndex + 1);
  testRegressorStatus.textContent =
    `Images left to test: ${left}; Average error so far: ${avgError.toFixed(1)} points`;
  testingIndex++;
  testingTimer = setTimeout(advanceTesting, 80);
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

// Loads a gallery image (supervised or unsupervised in origin) into the
// grading pane for a one-off reclassification. Interrupts any active
// review session, since both drive the same right-pane controls.
function startManualReclassify(imageId, score) {
  if (running || !currentModelId) return;
  stopReviewing();
  currentImageId = null;
  currentAutoImageId = null;
  manualImageId = imageId;
  manualReclassifying = true;
  updateGradingSectionVisibility();

  const initialScore = score != null ? score : 50;
  currentImage.src = `/api/image/${imageId}?session_id=${encodeURIComponent(currentModelId)}`;
  currentImage.classList.remove('hidden');
  noImageMessage.classList.add('hidden');
  setSliderValue(initialScore);
  renderJudgement(initialScore, 'Currently rated');
  setScoreControlsEnabled(true);
}

async function submitManualReclassify(score) {
  if (!manualImageId || !currentModelId) return;
  setScoreControlsEnabled(false);
  await fetch(`/api/models/${encodeURIComponent(currentModelId)}/promote-image`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: manualImageId, score }),
  });
  manualReclassifying = false;
  manualImageId = null;
  updateGradingSectionVisibility();
  galleryOffset = 0;
  await refreshGallery();
  await refreshSiteStats();
  await refreshScoreHistogram();
  await refreshModelList();
  if (currentModelId) modelSelect.value = currentModelId;
}

async function pollNextAutoImage() {
  if (!reviewingAuto || !currentModelId) return;
  const modelId = currentModelId;

  currentAutoImageId = null;
  currentImage.classList.add('hidden');
  noImageMessage.textContent = 'Loading next image to review…';
  noImageMessage.classList.remove('hidden');
  setJudgementNeutral('Loading…');
  setScoreControlsEnabled(false);

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
  setSliderValue(data.score);
  renderJudgement(data.score, 'This was auto-classified with a score of');
  setScoreControlsEnabled(true);
}

async function promoteCurrentAutoImage(score) {
  if (!currentAutoImageId || !currentModelId) return;
  setScoreControlsEnabled(false);
  await fetch(`/api/models/${encodeURIComponent(currentModelId)}/promote-image`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: currentAutoImageId, score }),
  });
  galleryOffset = 0;
  await refreshGallery();
  await refreshSiteStats();
  await refreshScoreHistogram();
  await refreshModelList();
  if (currentModelId) modelSelect.value = currentModelId;
  pollNextAutoImage();
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
  setSaveThresholdValue(s.save_threshold);
  if (s.last_error) {
    statusLastError.textContent = `Source error: ${s.last_error}`;
    statusLastError.classList.remove('hidden');
  } else {
    statusLastError.classList.add('hidden');
  }
  await refreshScoreHistogram();
}

async function pollNextImage() {
  if (!running || mode !== 'supervised' || !currentModelId) return;

  currentImageId = null;
  currentImage.classList.add('hidden');
  noImageMessage.textContent = 'Waiting for the next image…';
  noImageMessage.classList.remove('hidden');
  setJudgementNeutral('Waiting for image…');
  pageInfoRow.classList.add('hidden');
  setScoreControlsEnabled(false);

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
  setSliderValue(data.prediction ? data.prediction.score : 50);
  if (data.prediction) {
    renderJudgement(data.prediction.score, 'The model predicts a score of');
  } else {
    setJudgementNeutral('Not enough grades yet to make a prediction — keep grading!');
  }
  if (data.source_page_url) {
    pageInfoLink.textContent = data.source_page_url;
    pageInfoLink.href = data.source_page_url;
    pageInfoRow.classList.remove('hidden');
  }
  setScoreControlsEnabled(true);
}

async function skipCurrentPage() {
  if (!currentImageId || !currentModelId) return;
  skipPageBtn.disabled = true;
  await fetch('/api/skip-page', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: currentModelId, image_id: currentImageId }),
  });
  currentImageId = null;
  pollNextImage();
}

skipPageBtn.addEventListener('click', skipCurrentPage);

function setSliderValue(score) {
  const rounded = Math.round(score);
  scoreSlider.value = rounded;
  scoreSliderValue.textContent = rounded;
}

function setJudgementNeutral(text) {
  judgementText.className = '';
  judgementText.style.color = '';
  judgementText.textContent = text;
}

// Continuous red (0) -> green (100) interpolation, replacing the old
// binary not_part/positive color split now that the underlying value is a
// continuous score rather than a discrete class.
function scoreToColor(score) {
  const low = [217, 70, 63]; // matches the old "not part of class" red
  const high = [47, 158, 68]; // matches the old "good" green
  const t = Math.max(0, Math.min(1, score / 100));
  const rgb = low.map((c, i) => Math.round(c + (high[i] - c) * t));
  return `rgb(${rgb.join(',')})`;
}

function renderJudgement(score, leadIn) {
  const rounded = Math.round(score);
  const icon = document.createElement('span');
  icon.className = 'judgement-icon';
  icon.setAttribute('aria-hidden', 'true');
  icon.textContent = score >= 50 ? '✓' : '✗';

  const scoreEl = document.createElement('strong');
  scoreEl.textContent = `${rounded}`;

  const message = document.createElement('span');
  message.append(`${leadIn} `, scoreEl, ' / 100.');

  judgementText.className = '';
  judgementText.style.color = scoreToColor(score);
  judgementText.replaceChildren(icon, message);
}

function setScoreControlsEnabled(enabled) {
  scoreSlider.disabled = !enabled;
  submitScoreBtn.disabled = !enabled;
  deleteImageBtn.disabled = !enabled;
  skipPageBtn.disabled = !enabled;
}

async function gradeCurrentImage(score) {
  if (!currentImageId || !currentModelId) return;
  setScoreControlsEnabled(false);
  await fetch('/api/grade', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: currentModelId, image_id: currentImageId, score }),
  });
  galleryOffset = 0;
  await refreshGallery();
  await refreshSiteStats();
  await refreshScoreHistogram();
  await refreshModelList();
  if (currentModelId) modelSelect.value = currentModelId;
  pollNextImage();
}

// Renders a score distribution as a classic bar chart: SCORE_HISTOGRAM_BINS
// equal-width columns (one per integer score), each column's height scaled
// relative to the tallest bin so the shape of the distribution is visible,
// colored red (lowest score) to green (highest) by its position.
async function refreshScoreHistogram() {
  if (!currentModelId) return;
  const modelId = currentModelId;
  const resp = await fetch(`/api/models/${encodeURIComponent(modelId)}/score-histogram`);
  if (!resp.ok || currentModelId !== modelId) return;
  const histogram = await resp.json();
  if (currentModelId !== modelId) return;

  renderScoreHistogramBar(scoreHistogramHuman, histogram.human);
  renderScoreHistogramBar(scoreHistogramAuto, histogram.auto);
  scoreHistogramHumanCount.textContent = histogram.human.reduce((a, b) => a + b, 0);
  scoreHistogramAutoCount.textContent = histogram.auto.reduce((a, b) => a + b, 0);
}

function renderScoreHistogramBar(container, bins) {
  const maxCount = Math.max(...bins);
  container.replaceChildren();
  container.classList.toggle('empty', maxCount === 0);
  if (maxCount === 0) return;

  // Log-scaled (not linear) -- the lowest/highest-score bins tend to
  // dominate the count so heavily that a linear scale flattens every other
  // bin to near-invisible. log1p handles zero-count bins cleanly (maps to
  // 0 height) without a -Infinity from log(0).
  const logMax = Math.log1p(maxCount);
  const binWidth = 100 / SCORE_HISTOGRAM_BINS;
  bins.forEach((count, i) => {
    const column = document.createElement('div');
    column.className = 'score-histogram-segment';
    column.style.height = `${(Math.log1p(count) / logMax) * 100}%`;
    const midpoint = i * binWidth + binWidth / 2;
    column.style.backgroundColor = scoreToColor(midpoint);
    if (count > 0) {
      const rangeStart = Math.round(i * binWidth);
      const rangeEnd = Math.round((i + 1) * binWidth);
      column.title = `${count} image${count === 1 ? '' : 's'} scored ${rangeStart}-${rangeEnd}`;
    }
    container.appendChild(column);
  });
}

async function refreshSiteStats() {
  if (!currentModelId) return;
  const modelId = currentModelId;
  const resp = await fetch(`/api/models/${encodeURIComponent(modelId)}/site-stats`);
  if (!resp.ok || currentModelId !== modelId) return;
  const stats = await resp.json();
  if (currentModelId !== modelId) return;

  siteStatsCount.textContent = stats.length;
  siteStatsList.replaceChildren();
  for (const site of stats) {
    const row = document.createElement('div');
    row.className = 'site-stats-row';

    const domainEl = document.createElement('span');
    domainEl.className = 'site-stats-domain';
    domainEl.textContent = site.domain;

    const scoreEl = document.createElement('span');
    scoreEl.className = 'site-stats-score';
    scoreEl.textContent = Math.round(site.average_score);
    scoreEl.style.color = scoreToColor(site.average_score);

    const countEl = document.createElement('span');
    countEl.className = 'site-stats-image-count';
    countEl.textContent = `${site.image_count} image${site.image_count === 1 ? '' : 's'}`;

    row.append(domainEl, scoreEl, countEl);
    siteStatsList.appendChild(row);
  }

  populateDomainFilter(stats.map((s) => s.domain));
}

function populateDomainFilter(domains) {
  const previousValue = galleryFilterDomain.value;
  galleryFilterDomain.innerHTML = '<option value="">All</option>';
  for (const domain of domains) {
    const option = document.createElement('option');
    option.value = domain;
    option.textContent = domain;
    galleryFilterDomain.appendChild(option);
  }
  if (domains.includes(previousValue)) {
    galleryFilterDomain.value = previousValue;
  }
}

async function refreshGallery() {
  if (!currentModelId) return;
  const modelId = currentModelId;
  const params = new URLSearchParams();
  params.set('min_score', galleryScoreFilterMin.value);
  params.set('max_score', galleryScoreFilterMax.value);
  if (galleryFilterSource.value) params.set('source', galleryFilterSource.value);
  if (galleryFilterDomain.value) params.set('domain', galleryFilterDomain.value);
  const sortRadio = document.querySelector('#gallery-sort input[name="gallery-sort"]:checked');
  if (sortRadio) params.set('sort', sortRadio.value);
  params.set('offset', galleryOffset);
  const resp = await fetch(`/api/models/${encodeURIComponent(modelId)}/images?${params}`);
  if (!resp.ok || currentModelId !== modelId) return;
  const page = await resp.json();
  if (currentModelId !== modelId) return;

  const images = page.items;
  const total = page.total;

  galleryCount.textContent = total;
  const rangeStart = total === 0 ? 0 : galleryOffset + 1;
  const rangeEnd = Math.min(galleryOffset + GALLERY_PAGE_SIZE, total);
  galleryPageInfo.textContent = `${rangeStart}–${rangeEnd} of ${total}`;
  galleryPrevBtn.disabled = galleryOffset <= 0;
  galleryNextBtn.disabled = galleryOffset + GALLERY_PAGE_SIZE >= total;

  galleryGrid.classList.toggle('locked', running);
  galleryGrid.replaceChildren();
  for (const img of images) {
    const cell = document.createElement('div');
    cell.className = 'gallery-cell';
    cell.tabIndex = 0;
    cell.setAttribute('role', 'button');
    cell.title = 'Click to reclassify this image';

    const thumb = document.createElement('img');
    thumb.src = `/api/image/${img.image_id}?session_id=${encodeURIComponent(modelId)}`;
    thumb.alt = img.label;
    thumb.loading = 'lazy';

    const badge = document.createElement('span');
    badge.className = 'gallery-badge';
    if (img.score != null) {
      badge.style.backgroundColor = scoreToColor(img.score);
      badge.style.color = '#16171b';
    }
    const originText = img.label === 'auto' ? 'auto' : 'human';
    badge.textContent = img.score != null ? `${originText} · ${Math.round(img.score)}` : originText;

    cell.append(thumb, badge);
    cell.addEventListener('click', () => startManualReclassify(img.image_id, img.score));
    cell.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        startManualReclassify(img.image_id, img.score);
      }
    });
    galleryGrid.appendChild(cell);
  }
}
