const CONFIG = {
  equipment: [
    { id: 'hat', name: '帽子' },
    { id: 'gloves', name: 'グローブ' },
    { id: 'shoes', name: 'シューズ' },
    { id: 'bag', name: 'バッグ' },
    { id: 'badge', name: 'バッジ' },
    { id: 'hairpin', name: 'ヘアピン' },
    { id: 'charm', name: 'お守り' },
    { id: 'watch', name: '腕時計' },
    { id: 'necklace', name: 'ネックレス' },
  ],
  tierRequirements: {
    1: 1,
    2: 40,
    3: 45,
    4: 50,
    5: 55,
    6: 65,
    7: 65,
    8: 60,
    9: 50,
    10: 60,
  },
  defaultTargetPeople: {
    hat: 3,
    gloves: 2,
    shoes: 3,
    bag: 2,
    badge: 2,
    hairpin: 4,
    charm: 1,
    watch: 4,
    necklace: 2,
  },
  dropLocations: [],
  dropRates: {
    slotWeights: {
      primary: 1.0,
      secondary1: 1.0,
      secondary2: 1.0,
    },
    lowerTierOffsets: [2, 3],
    lowerTierWeights: {
      2: 0.30,
      3: 0.20,
    },
    priorityShortageRateWeight: 0.55,
    priorityShortageCountWeight: 0.35,
    priorityRatioWeight: 0.10,
  },
  learningMode: false,
  confidenceThreshold: {
    ocr: 0.5,
    equipment: 0.7,
    tier: 0.7,
  },
  correctionExportName: 'ba_corrections_export.json',
};

function loadUserConfig() {
  try {
    const saved = localStorage.getItem('ba_user_config');
    if (!saved) {
      return;
    }
    const parsed = JSON.parse(saved);
    CONFIG.learningMode = Boolean(parsed.learningMode);
  } catch (_err) {
    CONFIG.learningMode = false;
  }
}

function saveUserConfig() {
  localStorage.setItem('ba_user_config', JSON.stringify({
    learningMode: CONFIG.learningMode,
  }));
}

function getEquipmentName(id) {
  if (id === 'material') {
    return '素材';
  }
  if (id === 'unknown') {
    return '不明';
  }
  const eq = CONFIG.equipment.find((item) => item.id === id);
  return eq ? eq.name : id;
}

async function ensureDropLocationsLoaded() {
  if (Array.isArray(CONFIG.dropLocations) && CONFIG.dropLocations.length > 0) {
    return CONFIG.dropLocations;
  }

  const candidates = [
    'data/drop_locations.json',
  ];

  for (const path of candidates) {
    try {
      const response = await fetch(path, { cache: 'no-store' });
      if (!response.ok) {
        continue;
      }
      const payload = await response.json();
      if (Array.isArray(payload.locations) && payload.locations.length > 0) {
        CONFIG.dropLocations = payload.locations;
        return CONFIG.dropLocations;
      }
    } catch (_err) {
      // Try next candidate.
    }
  }

  CONFIG.dropLocations = [];
  return CONFIG.dropLocations;
}

loadUserConfig();
