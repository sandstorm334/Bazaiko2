const STORAGE_KEYS = {
  COUNTS: 'ba_equipment_counts',
  TARGET_PEOPLE: 'ba_target_people',
  THEME: 'ba_theme',
  CORRECTION_EXPORTS: 'ba_correction_exports',
};

function loadCounts() {
  try {
    const data = localStorage.getItem(STORAGE_KEYS.COUNTS);
    return data ? JSON.parse(data) : {};
  } catch (_err) {
    return {};
  }
}

function saveCounts(counts) {
  localStorage.setItem(STORAGE_KEYS.COUNTS, JSON.stringify(counts));
}

function loadTargetPeople() {
  try {
    const data = localStorage.getItem(STORAGE_KEYS.TARGET_PEOPLE);
    return data ? JSON.parse(data) : { ...CONFIG.defaultTargetPeople };
  } catch (_err) {
    return { ...CONFIG.defaultTargetPeople };
  }
}

function saveTargetPeople(targets) {
  localStorage.setItem(STORAGE_KEYS.TARGET_PEOPLE, JSON.stringify(targets));
}

function loadTheme() {
  return localStorage.getItem(STORAGE_KEYS.THEME) || 'light';
}

function saveTheme(theme) {
  localStorage.setItem(STORAGE_KEYS.THEME, theme);
}

function loadCorrectionExports() {
  try {
    const data = localStorage.getItem(STORAGE_KEYS.CORRECTION_EXPORTS);
    return data ? JSON.parse(data) : [];
  } catch (_err) {
    return [];
  }
}

function saveCorrectionExports(rows) {
  localStorage.setItem(STORAGE_KEYS.CORRECTION_EXPORTS, JSON.stringify(rows));
}

function appendCorrectionExport(entries) {
  const current = loadCorrectionExports();
  current.push(...entries);
  saveCorrectionExports(current);
}

function exportToCsv(counts) {
  const lines = ['equipment,tier,count'];
  for (const [eqId, tiers] of Object.entries(counts)) {
    for (const [tier, count] of Object.entries(tiers)) {
      lines.push(`${eqId},${tier},${count}`);
    }
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `equipment_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function importFromCsv(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = String(e.target.result || '');
        const lines = text.split(/\r?\n/).filter((line) => line.trim());
        const counts = {};
        for (let i = 1; i < lines.length; i++) {
          const [eqId, tier, count] = lines[i].split(',');
          if (!eqId || !tier) {
            continue;
          }
          if (!counts[eqId]) {
            counts[eqId] = {};
          }
          counts[eqId][tier] = parseInt(count, 10) || 0;
        }
        resolve(counts);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

function downloadJsonFile(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
