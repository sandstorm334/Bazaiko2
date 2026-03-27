let appState = {
  counts: {},
  targetPeople: {},
  theme: 'light',
  recommendations: [],
  peopleProgress: {},
};

let recognitionItems = [];
let selectedIndices = new Set();
let activeIndex = -1;
let correctionDirectoryHandle = null;
let latestRecognitionMetrics = null;
let modelReady = false;
let recognitionRunning = false;
let resumeRecoveryRunning = false;

const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => document.querySelectorAll(sel);
const isFileProtocol = () => window.location.protocol === 'file:';
const bindIfPresent = (selector, eventName, handler) => {
  const element = qs(selector);
  if (!element) {
    console.warn(`Missing element: ${selector}`);
    return null;
  }
  element.addEventListener(eventName, handler);
  return element;
};

function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    if (!canvas) {
      reject(new Error('canvas missing'));
      return;
    }
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error('canvas toBlob failed'));
      }
    }, 'image/png');
  });
}

async function ensureItemImageUrl(item, kind) {
  if (!item) {
    return '';
  }
  const urlKey = kind === 'count' ? 'countObjectUrl' : 'cropObjectUrl';
  const canvasKey = kind === 'count' ? 'countCanvas' : 'cropCanvas';
  if (item[urlKey]) {
    return item[urlKey];
  }
  if (!item[canvasKey]) {
    return '';
  }
  const blob = await canvasToBlob(item[canvasKey]);
  item[urlKey] = URL.createObjectURL(blob);
  return item[urlKey];
}

async function ensureItemDataUrl(item, kind) {
  const url = await ensureItemImageUrl(item, kind);
  if (!url) {
    return '';
  }
  const response = await fetch(url);
  const blob = await response.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(new Error('blob to data url failed'));
    reader.readAsDataURL(blob);
  });
}

function revokeRecognitionAssetUrls(items) {
  for (const item of items || []) {
    if (item.cropObjectUrl) {
      URL.revokeObjectURL(item.cropObjectUrl);
      item.cropObjectUrl = '';
    }
    if (item.countObjectUrl) {
      URL.revokeObjectURL(item.countObjectUrl);
      item.countObjectUrl = '';
    }
  }
}

async function init() {
  await ensureDropLocationsLoaded();
  appState.counts = loadCounts();
  appState.targetPeople = loadTargetPeople();
  appState.theme = loadTheme();
  applyTheme(appState.theme);
  setupEventListeners();
  refreshInventoryViews();
  const statusEl = qs('#status');
  const modelStatusEl = qs('#model-status');
  if (isFileProtocol()) {
    if (statusEl) {
      statusEl.textContent = 'file:// では認識機能は動作しません';
    }
    if (modelStatusEl) {
      modelStatusEl.textContent = 'start-webapp.bat または start-webapp.ps1 で起動してください';
    }
  } else {
    if (statusEl) {
      statusEl.textContent = 'モデル読込待機中';
    }
    if (modelStatusEl) {
      modelStatusEl.textContent = 'モデルをバックグラウンドで読み込み中';
    }
  }
}

function setupEventListeners() {
  qsa('.tab').forEach((tab) => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });
  bindIfPresent('#theme-toggle', 'click', toggleTheme);
  bindIfPresent('#recognition-files', 'change', handleFilesSelected);
  bindIfPresent('#run-recognition', 'click', runRecognition);
  bindIfPresent('#clear-selection', 'click', clearSelections);
  bindIfPresent('#open-corrections', 'click', async () => {
    await openCorrectionModal();
  });
  bindIfPresent('#pick-correction-dir', 'click', pickCorrectionDirectory);

  const learningToggle = qs('#learning-mode-toggle');
  if (learningToggle) {
    learningToggle.checked = CONFIG.learningMode;
    learningToggle.addEventListener('change', (event) => {
      CONFIG.learningMode = event.target.checked;
      saveUserConfig();
    });
  }

  bindIfPresent('#close-corrections', 'click', closeCorrectionModal);
  bindIfPresent('#apply-corrections', 'click', applyCorrections);
  bindIfPresent('#correction-modal', 'click', (event) => {
    if (event.target.id === 'correction-modal') {
      closeCorrectionModal();
    }
  });

  bindIfPresent('#open-target-modal', 'click', openTargetModal);
  bindIfPresent('#close-target-modal', 'click', closeTargetModal);
  bindIfPresent('#save-targets', 'click', saveTargets);
  bindIfPresent('#reset-targets', 'click', resetTargets);
  bindIfPresent('#target-modal', 'click', (event) => {
    if (event.target.id === 'target-modal') {
      closeTargetModal();
    }
  });

  document.addEventListener('keydown', handleKeydown);
  document.addEventListener('model-status', handleModelStatus);
  document.addEventListener('visibilitychange', handleVisibilityChange);
  window.addEventListener('pageshow', handlePageShow);
  window.addEventListener('focus', handleWindowFocus);
}

async function recoverAfterPageResume(reason) {
  if (isFileProtocol() || resumeRecoveryRunning) {
    return;
  }
  resumeRecoveryRunning = true;
  try {
    const statusEl = qs('#status');
    const modelStatusEl = qs('#model-status');
    if (!recognitionRunning && statusEl) {
      statusEl.textContent = 'ページ復帰を確認、状態を再同期中...';
    }
    if (modelStatusEl) {
      modelStatusEl.textContent = `モデル状態を再確認中 (${reason})`;
    }
    const ready = await ensureModelSessionsReady();
    if (!ready) {
      throw new Error('モデルセッションの復旧に失敗しました');
    }
    // UI freeze recovery: refresh stateful views on return.
    appState.counts = loadCounts();
    appState.targetPeople = loadTargetPeople();
    refreshInventoryViews();
    renderRecognitionTable();
    if (activeIndex >= 0) {
      await selectRow(activeIndex);
    }
    if (!recognitionRunning && statusEl) {
      statusEl.textContent = '準備完了';
    }
    if (modelStatusEl) {
      modelStatusEl.textContent = 'モデル準備完了 (復帰後再同期)';
    }
    recognitionRunning = false;
  } catch (error) {
    console.error('Resume recovery failed:', error);
    const statusEl = qs('#status');
    const modelStatusEl = qs('#model-status');
    if (statusEl) {
      statusEl.textContent = '復帰処理失敗';
    }
    if (modelStatusEl) {
      modelStatusEl.textContent = `復帰処理エラー: ${error.message}`;
    }
  } finally {
    resumeRecoveryRunning = false;
  }
}

function handleVisibilityChange() {
  if (document.visibilityState === 'visible') {
    void recoverAfterPageResume('visibilitychange');
  }
}

function handlePageShow(event) {
  if (event.persisted) {
    void recoverAfterPageResume('pageshow-bfcache');
  }
}

function handleWindowFocus() {
  if (document.visibilityState === 'visible') {
    void recoverAfterPageResume('focus');
  }
}

function handleModelStatus(event) {
  const detail = event.detail || {};
  const message = detail.message || '';
  modelReady = detail.state === 'ready';
  const modelStatusEl = qs('#model-status');
  const statusEl = qs('#status');
  if (modelStatusEl) {
    modelStatusEl.textContent = message || 'モデル状態不明';
  }
  if (!recognitionRunning && statusEl) {
    statusEl.textContent = modelReady ? '準備完了' : (message || 'モデル読み込み中');
  }
}

function switchTab(tabId) {
  qsa('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === tabId));
  qsa('.tab-panel').forEach((panel) => panel.classList.toggle('active', panel.id === tabId));
}

function getActiveTabId() {
  const active = document.querySelector('.tab.active');
  return active ? active.dataset.tab : 'inventory';
}

function applyTheme(theme) {
  document.body.classList.remove('light', 'dark');
  document.body.classList.add(theme);
  appState.theme = theme;
  qs('#theme-toggle').textContent = `テーマ: ${theme === 'dark' ? 'ダーク' : 'ライト'}`;
  saveTheme(theme);
}

function toggleTheme() {
  applyTheme(appState.theme === 'dark' ? 'light' : 'dark');
}

function ratioColor(ratio) {
  const clamped = Math.max(0, Math.min(ratio, 2));
  if (clamped <= 1) {
    const g = Math.round(30 + 225 * clamped);
    return `rgb(255, ${g}, ${g})`;
  }
  const t = clamped - 1;
  const r = Math.round(255 * (1 - t));
  return `rgb(${r}, 255, ${r})`;
}

function refreshInventoryViews() {
  appState.peopleProgress = BAWebLogic.calculatePeopleProgress(CONFIG, appState.counts, appState.targetPeople);
  appState.recommendations = BAWebLogic.recommendStages(CONFIG, appState.counts, appState.targetPeople, 8);
  buildTable();
  updateSummary();
}

function buildTable() {
  const wrap = qs('#table-wrap');
  const tiers = Object.keys(CONFIG.tierRequirements).sort((a, b) => Number(a) - Number(b));
  let html = '<table class="table"><thead><tr><th>Tier</th><th>必要数</th>';
  for (const eq of CONFIG.equipment) {
    const progress = appState.peopleProgress[eq.id] || { complete: 0, target: 0 };
    html += `<th>${eq.name}<br><small class="header-target">${progress.complete}/${progress.target}人分</small></th>`;
  }
  html += '</tr></thead><tbody>';

  for (const tier of tiers) {
    const req = CONFIG.tierRequirements[tier];
    html += `<tr><td class="header">T${tier}</td><td class="header">${req}</td>`;
    for (const eq of CONFIG.equipment) {
      const count = Number(((appState.counts[eq.id] || {})[tier]) || 0);
      const target = req * Number(appState.targetPeople[eq.id] || 0);
      const ratio = target > 0 ? count / target : 0;
      html += `
        <td class="data-cell" data-eq="${eq.id}" data-tier="${tier}" style="background:${ratioColor(ratio)}">
          <div class="cell-main">${count.toLocaleString()}</div>
          <div class="cell-sub">${count.toLocaleString()}/${target.toLocaleString()}</div>
        </td>`;
    }
    html += '</tr>';
  }

  html += '</tbody></table>';
  wrap.innerHTML = html;
  wrap.querySelectorAll('.data-cell').forEach((cell) => {
    cell.addEventListener('click', () => showDetail(cell.dataset.eq, Number(cell.dataset.tier)));
    cell.addEventListener('dblclick', () => editCell(cell));
  });
}

function showDetail(eqId, tier) {
  const name = getEquipmentName(eqId);
  const count = Number(((appState.counts[eqId] || {})[String(tier)]) || 0);
  const required = Number(CONFIG.tierRequirements[String(tier)] || 0);
  const targetPeople = Number(appState.targetPeople[eqId] || 0);
  const target = required * targetPeople;
  const missing = Math.max(0, target - count);
  const stages = BAWebLogic.getDropStages(CONFIG.dropLocations, eqId, tier);
  const mainDrops = stages.main.length ? stages.main.join(', ') : 'なし';
  const subDrops = stages.sub.length ? stages.sub.join(', ') : 'なし';
  const detail = [
    `${name} T${tier}`,
    `所持数: ${count.toLocaleString()}`,
    `目標数: ${target.toLocaleString()} (${targetPeople}人分)`,
    `不足: ${missing.toLocaleString()}`,
    `メインドロップ(記載Tier): ${mainDrops}`,
    `サブドロップ(下位Tier): ${subDrops}`,
  ];
  qs('#detail').textContent = detail.join('\n');
}

function editCell(cell) {
  const eqId = cell.dataset.eq;
  const tier = String(cell.dataset.tier);
  const current = Number(((appState.counts[eqId] || {})[tier]) || 0);
  const input = document.createElement('input');
  input.type = 'number';
  input.value = String(current);
  input.min = '0';
  input.className = 'count-edit-input';
  cell.innerHTML = '';
  cell.appendChild(input);
  input.focus();
  input.select();

  const commit = () => {
    if (!appState.counts[eqId]) {
      appState.counts[eqId] = {};
    }
    appState.counts[eqId][tier] = Math.max(0, Number(input.value || 0));
    saveCounts(appState.counts);
    refreshInventoryViews();
  };

  input.addEventListener('blur', commit);
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      commit();
    } else if (event.key === 'Escape') {
      refreshInventoryViews();
    }
  });
}

function updateSummary() {
  let totalCurrent = 0;
  let totalTarget = 0;
  let totalMissing = 0;
  const shortages = [];

  for (const eq of CONFIG.equipment) {
    for (const [tier, reqRaw] of Object.entries(CONFIG.tierRequirements)) {
      const req = Number(reqRaw);
      const count = Number(((appState.counts[eq.id] || {})[tier]) || 0);
      const target = req * Number(appState.targetPeople[eq.id] || 0);
      totalTarget += target;
      totalCurrent += Math.min(count, target);
      if (count < target) {
        shortages.push({
          equipment_id: eq.id,
          equipment_name: eq.name,
          tier: Number(tier),
          missing: target - count,
          ratio: target > 0 ? count / target : 0,
        });
        totalMissing += target - count;
      }
    }
  }

  const completion = totalTarget > 0 ? (totalCurrent / totalTarget) * 100 : 0;
  qs('#summary-completion').textContent = `${completion.toFixed(1)}%`;
  qs('#summary-missing').textContent = totalMissing.toLocaleString();
  renderRankList(qs('#shortage-count'), [...shortages].sort((a, b) => b.missing - a.missing), (item) => item.missing.toLocaleString());
  renderRankList(qs('#shortage-ratio'), [...shortages].sort((a, b) => a.ratio - b.ratio), (item) => `${(item.ratio * 100).toFixed(0)}%`);
  renderPeopleProgress();
  renderRecommendations();
}

function renderRankList(container, rows, valueFormatter) {
  container.innerHTML = rows.slice(0, 6).map((row, index) => `
    <div class="rank-row">
      <span>${index + 1}</span>
      <span>${row.equipment_name} T${row.tier}</span>
      <span>${valueFormatter(row)}</span>
    </div>
  `).join('') || '<p class="muted">データなし</p>';
}

function renderPeopleProgress() {
  const container = qs('#people-progress');
  container.innerHTML = CONFIG.equipment.map((eq) => {
    const progress = appState.peopleProgress[eq.id] || { complete: 0, target: 0, ratio: 0 };
    return `
      <div class="rank-row">
        <span>${eq.name}</span>
        <span>${progress.complete}/${progress.target}人</span>
        <span>${(progress.ratio * 100).toFixed(0)}%</span>
      </div>`;
  }).join('');
}

function renderRecommendations() {
  const container = qs('#recommend-list');
  if (!appState.recommendations.length) {
    container.innerHTML = '<p class="muted">おすすめなし</p>';
    return;
  }

  container.innerHTML = appState.recommendations.map((rec) => {
    const drops = rec.stageDrops
      .filter((drop) => drop.kind === 'main')
      .slice(0, 3)
      .map((drop) => `<span class="recommend-drop-tag main-drop">${getEquipmentName(drop.equipment_id)} T${drop.tier}</span>`)
      .join('');
    return `
      <div class="recommend-item">
        <div>
          <div class="recommend-stage">${rec.stageId}</div>
          <div class="recommend-meta">評価 ${rec.score.toFixed(2)}</div>
        </div>
        <div class="recommend-drops">${drops}</div>
        <div class="recommend-score">メインドロップ3種</div>
      </div>`;
  }).join('');
}

function handleFilesSelected(event) {
  const files = Array.from(event.target.files || []);
  qs('#file-list').innerHTML = files.map((file) => `<li>${file.name}</li>`).join('');
  qs('#status').textContent = `${files.length}ファイル選択済み`;
}

async function runRecognition() {
  if (isFileProtocol()) {
    alert('このページは file:// で開かれています。start-webapp.bat または start-webapp.ps1 で起動してください。');
    return;
  }
  const files = Array.from(qs('#recognition-files').files || []);
  if (!files.length) {
    alert('スクリーンショットを選択してください。');
    return;
  }

  revokeRecognitionAssetUrls(recognitionItems);
  recognitionItems = [];
  selectedIndices.clear();
  recognitionRunning = true;
  qs('#progress-bar').style.width = '0%';
  qs('#progress-meta').textContent = '0%';
  qs('#status').textContent = '認識開始';

  try {
    const result = await recognizeImagesInParallel(files, (progress, label) => {
      qs('#progress-bar').style.width = `${progress}%`;
      qs('#progress-meta').textContent = `${progress.toFixed(0)}%`;
      qs('#status').textContent = label;
    });
    recognitionItems = result.items;
    latestRecognitionMetrics = result.metrics;
    appState.counts = result.aggregatedCounts || {};
    saveCounts(appState.counts);
    refreshInventoryViews();
    renderRecognitionTable();
    qs('#recognition-metrics').textContent = formatRecognitionMetrics(result.metrics);
    if (recognitionItems.some((item) => item.status === 'needs_correction')) {
      await openCorrectionModal();
    }
  } catch (error) {
    console.error(error);
    alert(`認識に失敗しました: ${error.message}`);
    qs('#status').textContent = '認識失敗';
  } finally {
    recognitionRunning = false;
  }
}

function formatRecognitionMetrics(metrics) {
  if (!metrics) {
    return '';
  }
  return `合計 ${(metrics.totalMs / 1000).toFixed(2)}s / 検出 ${(metrics.detectMs / 1000).toFixed(2)}s / 分類 ${(metrics.classifyMs / 1000).toFixed(2)}s / OCR ${(metrics.ocrMs / 1000).toFixed(2)}s`;
}

function renderRecognitionTable() {
  const tbody = qs('#recognition-tbody');
  tbody.innerHTML = recognitionItems.map((item, index) => `
    <tr data-index="${index}" class="${item.status === 'needs_correction' ? 'status-needs_correction' : ''}">
      <td><input type="checkbox" class="select-check" ${selectedIndices.has(index) ? 'checked' : ''}></td>
      <td>${getEquipmentName(item.equipment_id)}</td>
      <td>${item.tier}</td>
      <td>${Number(item.count || 0).toLocaleString()}</td>
      <td class="status-${item.status}">${item.status === 'ok' ? 'OK' : item.status === 'corrected' ? '訂正済' : '要確認'}</td>
    </tr>
  `).join('');

  tbody.querySelectorAll('tr').forEach((row) => {
    row.addEventListener('click', (event) => {
      if (event.target.classList.contains('select-check')) {
        return;
      }
      void selectRow(Number(row.dataset.index));
    });
    row.querySelector('.select-check').addEventListener('change', () => {
      toggleSelection(Number(row.dataset.index));
    });
  });

  const corrections = recognitionItems.filter((item) => item.status === 'needs_correction');
  qs('#correction-list').innerHTML = corrections.length
    ? corrections.map((item) => `<div class="rank-row"><span>#${item.index + 1}</span><span>${getEquipmentName(item.equipment_id)} T${item.tier}</span><span>${item.count}</span></div>`).join('')
    : '<p class="muted">訂正候補なし</p>';
}

async function selectRow(index) {
  activeIndex = index;
  let activeRow = null;
  qsa('#recognition-tbody tr').forEach((row) => {
    const isActive = Number(row.dataset.index) === index;
    row.classList.toggle('active', isActive);
    if (isActive) {
      activeRow = row;
    }
  });
  const item = recognitionItems[index];
  if (!item) {
    return;
  }
  if (activeRow) {
    activeRow.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
  const [cropUrl, countUrl] = await Promise.all([
    ensureItemImageUrl(item, 'crop'),
    ensureItemImageUrl(item, 'count'),
  ]);
  if (activeIndex !== index) {
    return;
  }
  qs('#preview-crop').src = cropUrl;
  qs('#preview-count').src = countUrl;
}

function toggleSelection(index) {
  if (selectedIndices.has(index)) {
    selectedIndices.delete(index);
  } else {
    selectedIndices.add(index);
  }
  renderRecognitionTable();
}

function clearSelections() {
  selectedIndices.clear();
  renderRecognitionTable();
}

function buildCorrectionRows() {
  return (selectedIndices.size > 0
    ? [...selectedIndices].map((index) => recognitionItems[index])
    : recognitionItems.filter((item) => item.status === 'needs_correction')
  );
}

async function openCorrectionModal() {
  const items = buildCorrectionRows();
  const container = qs('#correction-modal-list');
  if (!items.length) {
    container.innerHTML = '<p class="muted">訂正対象はありません。</p>';
  } else {
    const urls = await Promise.all(items.map(async (item) => ({
      crop: await ensureItemImageUrl(item, 'crop'),
      count: await ensureItemImageUrl(item, 'count'),
    })));
    container.innerHTML = items.map((item, index) => `
      <div class="correction-row" data-index="${item.index}">
        <div>#${item.index + 1}</div>
        <img src="${urls[index].crop}" alt="crop">
        <img src="${urls[index].count}" alt="count">
        <select class="correction-equipment">
          ${CONFIG.equipment.map((eq) => `<option value="${eq.id}" ${item.equipment_id === eq.id ? 'selected' : ''}>${eq.name}</option>`).join('')}
          <option value="material" ${item.equipment_id === 'material' ? 'selected' : ''}>素材</option>
          <option value="unknown" ${item.equipment_id === 'unknown' ? 'selected' : ''}>不明</option>
        </select>
        <input type="number" class="correction-tier" value="${item.tier}" min="0" max="10">
        <input type="number" class="correction-count" value="${item.count}" min="0">
      </div>
    `).join('');
    container.querySelectorAll('.correction-equipment').forEach((select) => {
      select.addEventListener('change', () => {
        if (select.value === 'material') {
          select.closest('.correction-row').querySelector('.correction-tier').value = '0';
        }
      });
    });
  }
  qs('#correction-modal').classList.remove('hidden');
}

function closeCorrectionModal() {
  qs('#correction-modal').classList.add('hidden');
}

async function applyCorrections() {
  const activeTabId = getActiveTabId();
  const activeRowIndex = activeIndex;
  const rows = [...qsa('#correction-modal-list .correction-row')];
  const exportEntries = [];

  for (const row of rows) {
    const index = Number(row.dataset.index);
    const item = recognitionItems[index];
    if (!item) {
      continue;
    }
    item.equipment_id = row.querySelector('.correction-equipment').value;
    item.tier = Number(row.querySelector('.correction-tier').value);
    item.count = Number(row.querySelector('.correction-count').value);
    item.status = 'corrected';
    item.ocrSource = item.ocrSource || 'manual';
    exportEntries.push(await buildCorrectionExportEntry(item));
  }

  const aggregated = BAWebLogic.aggregateRecognitionItems(CONFIG, recognitionItems);
  appState.counts = aggregated.counts;
  saveCounts(appState.counts);
  refreshInventoryViews();
  renderRecognitionTable();
  closeCorrectionModal();
  switchTab(activeTabId);
  if (activeRowIndex >= 0) {
    await selectRow(activeRowIndex);
  }
  if (exportEntries.length > 0) {
    await persistCorrectionsToDirectory(exportEntries);
  }
  switchTab(activeTabId);
}

async function buildCorrectionExportEntry(item) {
  return {
    equipment_id: item.equipment_id,
    tier: item.tier,
    count: item.count,
    file_name: item.fileName,
    created_at: new Date().toISOString(),
    crop_data_url: await ensureItemDataUrl(item, 'count'),
  };
}

async function pickCorrectionDirectory() {
  if (!window.showDirectoryPicker) {
    alert('このブラウザはディレクトリ保存 API に未対応です。');
    return;
  }
  try {
    correctionDirectoryHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
    alert(`保存先を設定しました: ${correctionDirectoryHandle.name}`);
  } catch (error) {
    if (error && error.name !== 'AbortError') {
      alert(`保存先設定に失敗しました: ${error.message}`);
    }
  }
}

async function persistCorrectionsToDirectory(entries) {
  if (!correctionDirectoryHandle || entries.length === 0) {
    return;
  }
  try {
    for (const entry of entries) {
      const equipmentDir = await correctionDirectoryHandle.getDirectoryHandle(entry.equipment_id, { create: true });
      const tierDir = await equipmentDir.getDirectoryHandle(`tier_${entry.tier}`, { create: true });
      const stamp = entry.created_at.replace(/[-:.TZ]/g, '').slice(0, 14);
      const baseName = `${entry.equipment_id}_T${entry.tier}_x${entry.count}_${stamp}`;
      const pngHandle = await tierDir.getFileHandle(`${baseName}.png`, { create: true });
      const textHandle = await tierDir.getFileHandle(`${baseName}.png.txt`, { create: true });
      const pngWriter = await pngHandle.createWritable();
      await pngWriter.write(await (await fetch(entry.crop_data_url)).blob());
      await pngWriter.close();
      const textWriter = await textHandle.createWritable();
      await textWriter.write(String(entry.count));
      await textWriter.close();
    }
  } catch (error) {
    console.error(error);
    alert(`補正データの保存に失敗しました: ${error.message}`);
  }
}

function handleKeydown(event) {
  const recognitionTabActive = qs('#recognition').classList.contains('active');
  if (!recognitionTabActive) {
    return;
  }
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) {
    return;
  }
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault();
    const next = Math.max(0, Math.min(recognitionItems.length - 1, activeIndex + (event.key === 'ArrowDown' ? 1 : -1)));
    void selectRow(next);
  } else if (event.key === 'Enter' && activeIndex >= 0) {
    event.preventDefault();
    toggleSelection(activeIndex);
  }
}

function openTargetModal() {
  const container = qs('#target-people-list');
  container.innerHTML = CONFIG.equipment.map((eq) => `
    <div class="target-row">
      <label>${eq.name}</label>
      <input type="number" class="target-input" data-eid="${eq.id}" value="${appState.targetPeople[eq.id] || 0}" min="0">
      <span class="target-default">(既定: ${CONFIG.defaultTargetPeople[eq.id] || 0})</span>
    </div>
  `).join('');
  qs('#target-modal').classList.remove('hidden');
}

function closeTargetModal() {
  qs('#target-modal').classList.add('hidden');
}

function saveTargets() {
  qsa('#target-people-list .target-input').forEach((input) => {
    appState.targetPeople[input.dataset.eid] = Math.max(0, Number(input.value || 0));
  });
  saveTargetPeople(appState.targetPeople);
  refreshInventoryViews();
  closeTargetModal();
}

function resetTargets() {
  qsa('#target-people-list .target-input').forEach((input) => {
    input.value = CONFIG.defaultTargetPeople[input.dataset.eid] || 0;
  });
}

document.addEventListener('DOMContentLoaded', () => {
  init().catch((error) => {
    console.error('Initialization failed:', error);
    const statusEl = qs('#status');
    const modelStatusEl = qs('#model-status');
    if (statusEl) {
      statusEl.textContent = '初期化失敗';
    }
    if (modelStatusEl) {
      modelStatusEl.textContent = `初期化エラー: ${error.message}`;
    }
  });
});
