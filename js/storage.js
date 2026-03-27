const STORAGE_KEYS = {
  COUNTS: 'ba_equipment_counts',
  TARGET_PEOPLE: 'ba_target_people',
  THEME: 'ba_theme',
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
