(function (root, factory) {
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = factory();
    return;
  }
  root.BAWebLogic = factory();
})(typeof self !== 'undefined' ? self : globalThis, function () {
  'use strict';

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function isValidEquipmentId(config, equipmentId) {
    return Boolean(config.equipment.find((item) => item.id === equipmentId));
  }

  function isValidTier(tier) {
    return Number.isInteger(tier) && tier >= 1 && tier <= 10;
  }

  function buildTargetCounts(config, targetPeople) {
    const counts = {};
    for (const eq of config.equipment) {
      counts[eq.id] = {};
      const people = Number(targetPeople[eq.id] || 0);
      for (const [tier, req] of Object.entries(config.tierRequirements)) {
        counts[eq.id][tier] = Number(req) * people;
      }
    }
    return counts;
  }

  function calculatePeopleProgress(config, counts, targetPeople) {
    const result = {};
    for (const eq of config.equipment) {
      const eqCounts = counts[eq.id] || {};
      const target = Number(targetPeople[eq.id] || 0);
      let minPeople = Number.POSITIVE_INFINITY;
      let progressPoints = 0;
      let totalPoints = 0;

      for (const [tier, required] of Object.entries(config.tierRequirements)) {
        const need = Math.max(1, Number(required));
        const have = Number(eqCounts[tier] || 0);
        minPeople = Math.min(minPeople, Math.floor(have / need));
        progressPoints += Math.min(have / need, target || 0);
        totalPoints += target || 0;
      }

      if (!Number.isFinite(minPeople)) {
        minPeople = 0;
      }

      result[eq.id] = {
        complete: Math.max(0, Math.min(minPeople, target)),
        target,
        ratio: totalPoints > 0 ? progressPoints / totalPoints : 0,
      };
    }
    return result;
  }

  function getDropStages(dropLocations, equipmentId, tier) {
    const main = [];
    const sub = [];
    for (const loc of dropLocations) {
      const slotIndex = Array.isArray(loc.slots) ? loc.slots.indexOf(equipmentId) : -1;
      if (slotIndex < 0) {
        continue;
      }
      const baseTier = Number(loc.base_tier);
      if (baseTier === Number(tier)) {
        main.push(loc.stage);
      } else if (baseTier - Number(tier) === 2 || baseTier - Number(tier) === 3) {
        sub.push(loc.stage);
      }
    }
    return { main, sub, all: [...main, ...sub] };
  }

  function getStageDrops(dropLocations, stageId, dropRates) {
    const loc = dropLocations.find((item) => item.stage === stageId);
    if (!loc) {
      return [];
    }
    const drops = [];
    const offsets = Array.isArray(dropRates.lowerTierOffsets) ? dropRates.lowerTierOffsets : [];
    const lowerWeights = dropRates.lowerTierWeights || {};
    loc.slots.forEach((equipmentId, index) => {
      if (!equipmentId) {
        return;
      }
      const slot = index === 0 ? 'primary' : (index === 1 ? 'secondary1' : 'secondary2');
      drops.push({
        equipment_id: equipmentId,
        tier: Number(loc.base_tier),
        slot,
        kind: 'main',
        weight: Number((dropRates.slotWeights || {})[slot] || 1),
      });
      for (const offset of offsets) {
        const offsetNum = Number(offset);
        const tier = Number(loc.base_tier) - offsetNum;
        if (tier <= 0) {
          continue;
        }
        drops.push({
          equipment_id: equipmentId,
          tier,
          slot: `${slot}_lower_${offsetNum}`,
          kind: 'sub',
          weight: Number((dropRates.slotWeights || {})[slot] || 1) * Number(lowerWeights[String(offsetNum)] || 0),
        });
      }
    });
    return drops;
  }

  function buildStageExpectation(dropLocations, dropRates) {
    const stageExpectation = new Map();
    for (const loc of dropLocations) {
      const expected = new Map();
      for (const drop of getStageDrops(dropLocations, loc.stage, dropRates)) {
        const key = `${drop.equipment_id}:${drop.tier}`;
        expected.set(key, (expected.get(key) || 0) + Number(drop.weight || 0));
      }
      stageExpectation.set(loc.stage, expected);
    }
    return stageExpectation;
  }

  function buildPriority(currentCounts, targetCounts, dropRates) {
    const rateWeight = Number(dropRates.priorityShortageRateWeight || 0.55);
    const countWeight = Number(dropRates.priorityShortageCountWeight || 0.35);
    const ratioWeight = Number(dropRates.priorityRatioWeight || 0.10);
    let maxShortage = 1;

    for (const [equipmentId, tiers] of Object.entries(targetCounts)) {
      for (const [tier, target] of Object.entries(tiers)) {
        const current = Number(((currentCounts[equipmentId] || {})[tier]) || 0);
        maxShortage = Math.max(maxShortage, Math.max(0, Number(target) - current));
      }
    }

    const priority = new Map();
    for (const [equipmentId, tiers] of Object.entries(targetCounts)) {
      for (const [tier, targetRaw] of Object.entries(tiers)) {
        const target = Number(targetRaw || 0);
        if (target <= 0) {
          continue;
        }
        const current = Number(((currentCounts[equipmentId] || {})[tier]) || 0);
        const shortage = Math.max(0, target - current);
        const shortageRate = shortage / Math.max(1, target);
        const shortageCount = shortage / maxShortage;
        const ratio = current / Math.max(1, target);
        const score = (
          rateWeight * shortageRate +
          countWeight * shortageCount +
          ratioWeight * (1 / (1 + ratio))
        );
        priority.set(`${equipmentId}:${tier}`, Math.max(0, score));
      }
    }
    return priority;
  }

  function recommendStages(config, currentCounts, targetPeople, maxResults) {
    const targetCounts = buildTargetCounts(config, targetPeople);
    const stageExpectation = buildStageExpectation(config.dropLocations, config.dropRates);
    const priority = buildPriority(currentCounts, targetCounts, config.dropRates);
    const ranked = [];

    for (const loc of config.dropLocations) {
      const expected = stageExpectation.get(loc.stage) || new Map();
      let score = 0;
      const matched = [];
      for (const [key, value] of expected.entries()) {
        const itemPriority = Number(priority.get(key) || 0);
        if (itemPriority <= 0 || value <= 0) {
          continue;
        }
        score += itemPriority * value;
        const [equipment_id, tierStr] = key.split(':');
        matched.push({
          equipment_id,
          tier: Number(tierStr),
          value: itemPriority * value,
          expectedRunsValue: value,
        });
      }
      if (score <= 0) {
        continue;
      }
      matched.sort((a, b) => b.value - a.value);
      ranked.push({
        stageId: loc.stage,
        score,
        items: matched.slice(0, 6),
        stageDrops: getStageDrops(config.dropLocations, loc.stage, config.dropRates),
      });
    }

    ranked.sort((a, b) => b.score - a.score);
    return ranked.slice(0, maxResults);
  }

  function sourceWeight(source) {
    if (source === 'template' || source === 'cnn') {
      return 1.12;
    }
    if (source === 'knn' || source === 'rapid_easy_fast') {
      return 1.06;
    }
    if (source === 'ocr' || source === 'rapidocr' || source === 'easyocr' || source === 'crnn_ocr' || source === 'crnn_ctc' || source === 'crnn_ocr_beam') {
      return 1.0;
    }
    return 0.96;
  }

  function statusPriority(status) {
    if (status === 'corrected') {
      return 3;
    }
    if (status === 'ok') {
      return 2;
    }
    if (status === 'needs_correction') {
      return 1;
    }
    return 0;
  }

  function voteWeight(item) {
    const confidence = clamp(Number(item.ocrConfidence ?? item.confidence ?? 0), 0, 1);
    let base = 1.0;
    if (item.status === 'corrected') {
      base = 5.0;
    } else if (item.status === 'ok') {
      base = 1.8;
    } else if (item.status === 'needs_correction') {
      base = 0.55;
    }
    let weight = base * (0.30 + confidence) * sourceWeight(item.ocrSource || item.source || 'ocr');
    if (Number(item.count || 0) === 0 && item.status !== 'corrected') {
      weight *= 0.25;
    }
    return weight;
  }

  function groupIsAmbiguous(group, voteByValue, selected) {
    if (!group.length) {
      return false;
    }
    if (group.some((item) => item.status === 'corrected')) {
      return false;
    }
    if (group.every((item) => item.status === 'needs_correction')) {
      return true;
    }

    const selectedVote = Number(voteByValue.get(selected) || 0);
    if (selectedVote <= 0) {
      return true;
    }

    let strongestOther = 0;
    for (const [value, vote] of voteByValue.entries()) {
      if (value !== selected) {
        strongestOther = Math.max(strongestOther, Number(vote));
      }
    }
    if (strongestOther >= selectedVote * 0.90) {
      return true;
    }

    const zeroVote = Number(voteByValue.get(0) || 0);
    let positiveVote = 0;
    for (const [value, vote] of voteByValue.entries()) {
      if (Number(value) > 0) {
        positiveVote = Math.max(positiveVote, Number(vote));
      }
    }
    if (selected === 0 && positiveVote >= selectedVote * 0.55) {
      return true;
    }
    if (selected > 0 && zeroVote >= selectedVote * 0.85) {
      return true;
    }
    return false;
  }

  function aggregateRecognitionItems(config, items) {
    const grouped = new Map();
    const conflicts = [];
    const conflictSet = new Set();
    const counts = {};

    function markConflict(item) {
      if (!conflictSet.has(item.index)) {
        conflicts.push(item);
        conflictSet.add(item.index);
      }
    }

    for (const item of items) {
      if (item.equipment_id === 'material') {
        if (item.status === 'needs_correction') {
          markConflict(item);
        }
        continue;
      }
      if (!isValidEquipmentId(config, item.equipment_id) || !isValidTier(Number(item.tier))) {
        markConflict(item);
        continue;
      }
      const key = `${item.equipment_id}:${item.tier}`;
      if (!grouped.has(key)) {
        grouped.set(key, []);
      }
      grouped.get(key).push(item);
      if (item.status === 'needs_correction') {
        markConflict(item);
      }
    }

    for (const [key, group] of grouped.entries()) {
      const voteByValue = new Map();
      const freq = new Map();
      const confSum = new Map();
      const statusMax = new Map();
      const hasCorrected = group.some((item) => item.status === 'corrected');

      for (const item of group) {
        const count = Number(item.count || 0);
        freq.set(count, (freq.get(count) || 0) + 1);
        confSum.set(count, (confSum.get(count) || 0) + Number(item.ocrConfidence ?? item.confidence ?? 0));
        voteByValue.set(count, (voteByValue.get(count) || 0) + voteWeight(item));
        statusMax.set(count, Math.max(statusMax.get(count) || 0, statusPriority(item.status)));
      }

      const selected = [...voteByValue.keys()].sort((a, b) => {
        const scoreA = [statusMax.get(a) || 0, voteByValue.get(a) || 0, confSum.get(a) || 0, freq.get(a) || 0];
        const scoreB = [statusMax.get(b) || 0, voteByValue.get(b) || 0, confSum.get(b) || 0, freq.get(b) || 0];
        for (let i = 0; i < scoreA.length; i++) {
          if (scoreA[i] !== scoreB[i]) {
            return scoreB[i] - scoreA[i];
          }
        }
        return 0;
      })[0];

      const ambiguous = groupIsAmbiguous(group, voteByValue, selected);
      if (ambiguous) {
        for (const item of group) {
          if (item.status !== 'corrected') {
            item.status = 'needs_correction';
            markConflict(item);
          }
        }
      } else if (!hasCorrected) {
        for (const item of group) {
          if (Number(item.count || 0) !== selected && item.status !== 'corrected') {
            item.status = 'needs_correction';
            markConflict(item);
          }
        }
      }

      const [equipmentId, tier] = key.split(':');
      if (!counts[equipmentId]) {
        counts[equipmentId] = {};
      }
      counts[equipmentId][tier] = selected;
    }

    return { counts, conflicts };
  }

  return {
    aggregateRecognitionItems,
    buildPriority,
    buildStageExpectation,
    buildTargetCounts,
    calculatePeopleProgress,
    getDropStages,
    getStageDrops,
    isValidEquipmentId,
    isValidTier,
    recommendStages,
  };
});
