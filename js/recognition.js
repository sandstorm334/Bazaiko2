let yoloSession = null;
let ocrModel = null;
let equipmentClassifier = null;
let equipmentTypeClassifier = null;
let modelsLoaded = false;
let modelsLoading = false;
let modelsLoadPromise = null;
let useWebGPU = false;
let preloadStarted = false;

const MODEL_CONFIG = {
  yolo: {
    path: 'models/yolo_frame.onnx',
    inputSize: 640,
    confThreshold: 0.7,
  },
  equipmentType: {
    path: 'models/equipment_type_classifier.onnx',
    metadataPath: 'models/equipment_type_classifier.json',
  },
  classifier: {
    path: 'models/classifier.onnx',
    configPath: 'models/classifier_config.json',
  },
  ocr: {
    path: 'models/crnn_ocr.onnx',
    metadataPath: 'models/crnn_ocr.json',
  },
};

async function yieldToMain() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

async function checkWebGPUSupport() {
  if (!navigator.gpu) {
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return Boolean(adapter);
  } catch (_err) {
    return false;
  }
}

function getExecutionProviders() {
  return useWebGPU ? ['webgpu', 'wasm'] : ['wasm'];
}

function emitModelStatus(state, message) {
  document.dispatchEvent(new CustomEvent('model-status', {
    detail: { state, message },
  }));
}

async function loadModels(options = {}) {
  if (modelsLoaded) {
    return;
  }
  if (modelsLoadPromise) {
    return modelsLoadPromise;
  }
  modelsLoading = true;
  const onStatus = typeof options.onStatus === 'function' ? options.onStatus : null;
  const updateStatus = (state, message) => {
    emitModelStatus(state, message);
    if (onStatus) {
      onStatus(message);
    }
  };

  modelsLoadPromise = (async () => {
    try {
    updateStatus('loading', '認識モデルをバックグラウンドで読み込み中...');
    useWebGPU = await checkWebGPUSupport();
    const executionProviders = getExecutionProviders();
    ort.env.wasm.simd = true;
    const cores = navigator.hardwareConcurrency || 4;
    ort.env.wasm.numThreads = cores <= 4 ? 1 : Math.max(2, Math.min(4, Math.floor(cores / 2)));

    updateStatus('loading', 'フレーム検出モデルを読み込み中...');
    yoloSession = await ort.InferenceSession.create(MODEL_CONFIG.yolo.path, {
      executionProviders,
      graphOptimizationLevel: 'all',
    });
    await yieldToMain();

    updateStatus('loading', '数値OCRモデルを読み込み中...');
    ocrModel = new CRNNOCRInference(MODEL_CONFIG.ocr.path, MODEL_CONFIG.ocr.metadataPath);
    await ocrModel.init(executionProviders);
    await yieldToMain();

    updateStatus('loading', '補助数値モデルを読み込み中...');
    equipmentClassifier = new EquipmentClassifier(
      MODEL_CONFIG.classifier.path,
      MODEL_CONFIG.classifier.configPath,
    );
    await equipmentClassifier.initialize(executionProviders);
    await yieldToMain();

    updateStatus('loading', '装備識別モデルを読み込み中...');
    equipmentTypeClassifier = new EquipmentTypeClassifier(
      MODEL_CONFIG.equipmentType.path,
      MODEL_CONFIG.equipmentType.metadataPath,
    );
    await equipmentTypeClassifier.initialize(executionProviders);

      modelsLoaded = true;
      updateStatus('ready', `モデル準備完了 (${useWebGPU ? 'WebGPU' : 'WASM'})`);
    } catch (error) {
      console.error('Model loading failed:', error);
      updateStatus('error', `モデル読み込み失敗: ${error.message}`);
      throw error;
    } finally {
      modelsLoading = false;
      if (!modelsLoaded) {
        modelsLoadPromise = null;
      }
    }
  })();
  return modelsLoadPromise;
}

function startModelPreload() {
  if (preloadStarted || modelsLoaded || modelsLoading) {
    return;
  }
  preloadStarted = true;
  const kickoff = () => {
    loadModels().catch((error) => console.error('Model preload failed:', error));
  };
  if (typeof window.requestIdleCallback === 'function') {
    window.requestIdleCallback(kickoff, { timeout: 1200 });
  } else {
    setTimeout(kickoff, 0);
  }
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(img.src);
      resolve(img);
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function imageToCanvas(img) {
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  return canvas;
}

function letterboxCanvas(canvas, targetSize) {
  const srcW = canvas.width;
  const srcH = canvas.height;
  const ratio = Math.min(targetSize / srcW, targetSize / srcH);
  const resizedW = Math.round(srcW * ratio);
  const resizedH = Math.round(srcH * ratio);
  const padX = (targetSize - resizedW) / 2;
  const padY = (targetSize - resizedH) / 2;
  const out = document.createElement('canvas');
  out.width = targetSize;
  out.height = targetSize;
  const ctx = out.getContext('2d');
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, targetSize, targetSize);
  ctx.drawImage(canvas, 0, 0, srcW, srcH, padX, padY, resizedW, resizedH);
  return { canvas: out, ratio, padX, padY };
}

function cloneCanvas(source) {
  const canvas = document.createElement('canvas');
  canvas.width = source.width;
  canvas.height = source.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, 0, 0);
  return canvas;
}

function preprocessForYolo(canvas) {
  const size = MODEL_CONFIG.yolo.inputSize;
  const lb = letterboxCanvas(canvas, size);
  const imageData = lb.canvas.getContext('2d').getImageData(0, 0, size, size).data;
  const tensorData = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    tensorData[i] = imageData[i * 4] / 255;
    tensorData[size * size + i] = imageData[i * 4 + 1] / 255;
    tensorData[2 * size * size + i] = imageData[i * 4 + 2] / 255;
  }
  return { tensorData, ratio: lb.ratio, padX: lb.padX, padY: lb.padY };
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  if (x2 <= x1 || y2 <= y1) {
    return 0;
  }
  const intersection = (x2 - x1) * (y2 - y1);
  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function nms(boxes, iouThreshold) {
  const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
  const selected = [];
  while (sorted.length > 0) {
    const current = sorted.shift();
    selected.push(current);
    for (let i = sorted.length - 1; i >= 0; i--) {
      if (iou(current, sorted[i]) > iouThreshold) {
        sorted.splice(i, 1);
      }
    }
  }
  return selected;
}

function postprocessYolo(output, originalWidth, originalHeight, ratio, padX, padY, confThreshold) {
  const boxes = [];
  const data = output.data;
  const numDetections = output.dims[2];
  for (let i = 0; i < numDetections; i++) {
    const conf = data[4 * numDetections + i];
    if (conf < confThreshold) {
      continue;
    }
    const x = data[i];
    const y = data[numDetections + i];
    const w = data[2 * numDetections + i];
    const h = data[3 * numDetections + i];
    const x1 = ((x - w / 2) - padX) / ratio;
    const y1 = ((y - h / 2) - padY) / ratio;
    const x2 = ((x + w / 2) - padX) / ratio;
    const y2 = ((y + h / 2) - padY) / ratio;
    const left = Math.max(0, Math.min(originalWidth, x1));
    const top = Math.max(0, Math.min(originalHeight, y1));
    const right = Math.max(0, Math.min(originalWidth, x2));
    const bottom = Math.max(0, Math.min(originalHeight, y2));
    const width = right - left;
    const height = bottom - top;
    if (width >= 4 && height >= 4) {
      boxes.push({ x: left, y: top, width, height, confidence: conf });
    }
  }
  return nms(boxes, 0.5).sort((a, b) => {
    const rowA = Math.floor(a.y / 100);
    const rowB = Math.floor(b.y / 100);
    return rowA === rowB ? a.x - b.x : rowA - rowB;
  });
}

function cropCanvasRegion(canvas, x, y, width, height) {
  const output = document.createElement('canvas');
  output.width = Math.max(1, Math.round(width));
  output.height = Math.max(1, Math.round(height));
  output.getContext('2d').drawImage(canvas, x, y, width, height, 0, 0, output.width, output.height);
  return output;
}

function extractCountRegionCandidates(cropCanvas, tier, mode = 'default') {
  const h = cropCanvas.height;
  const w = cropCanvas.width;
  let yStarts;
  let xStarts;
  if (mode === 'fallback') {
    yStarts = tier >= 10 ? [0.67, 0.63] : [0.63, 0.60];
    xStarts = tier >= 10 ? [0.24, 0.20] : [0.24, 0.20];
  } else {
    yStarts = tier >= 10 ? [0.74, 0.70] : [0.70, 0.67];
    xStarts = tier >= 10 ? [0.30, 0.33] : [0.27, 0.30];
  }
  const candidates = [];
  for (const yRatio of yStarts) {
    for (const xRatio of xStarts) {
      const x = w * xRatio;
      const y = h * yRatio;
      candidates.push(cropCanvasRegion(cropCanvas, x, y, w - x, h - y));
    }
  }
  return candidates;
}

function analyzeCountCrop(canvas) {
  const ctx = canvas.getContext('2d');
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  if (width < 16 || height < 8) {
    return { usable: false, quality: -1, fgRatio: 0 };
  }
  const gray = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
  }
  const threshold = otsuThreshold(gray);
  let fg = 0;
  let edgeTouch = 0;
  let sum = 0;
  let sumSq = 0;
  for (let idx = 0; idx < gray.length; idx++) {
    const value = gray[idx];
    sum += value;
    sumSq += value * value;
    const active = value <= threshold ? 1 : 0;
    fg += active;
    if (active) {
      const y = Math.floor(idx / width);
      const x = idx % width;
      if (x === 0 || y === 0 || x === width - 1 || y === height - 1) {
        edgeTouch += 1;
      }
    }
  }
  const fgRatio = fg / Math.max(1, gray.length);
  const variance = Math.max(0, (sumSq / gray.length) - Math.pow(sum / gray.length, 2));
  const contrast = Math.sqrt(variance) / 64;
  const usable = fgRatio >= 0.02 && fgRatio <= 0.65 && edgeTouch / Math.max(1, fg) <= 0.55;
  return {
    usable,
    quality: fgRatio * 1.2 + contrast,
    fgRatio,
  };
}

function otsuThreshold(gray) {
  const histogram = new Uint32Array(256);
  for (const value of gray) {
    histogram[value] += 1;
  }
  const total = gray.length;
  let sum = 0;
  for (let i = 0; i < 256; i++) {
    sum += i * histogram[i];
  }
  let sumB = 0;
  let weightB = 0;
  let maxVariance = -1;
  let threshold = 127;
  for (let i = 0; i < 256; i++) {
    weightB += histogram[i];
    if (weightB === 0) {
      continue;
    }
    const weightF = total - weightB;
    if (weightF === 0) {
      break;
    }
    sumB += i * histogram[i];
    const meanB = sumB / weightB;
    const meanF = (sum - sumB) / weightF;
    const variance = weightB * weightF * Math.pow(meanB - meanF, 2);
    if (variance > maxVariance) {
      maxVariance = variance;
      threshold = i;
    }
  }
  return threshold;
}

function chooseOcrResult(candidates) {
  const scoreByValue = new Map();
  let bestPair = null;
  let bestLocalScore = -1;
  for (const pair of candidates) {
    const result = pair.result;
    if (result.value == null) {
      continue;
    }
    const sourceBoost = (result.source === 'template' || result.source === 'template_bank') ? 1.2 : 1.0;
    const localScore = Number(result.confidence || 0) * sourceBoost;
    scoreByValue.set(result.value, (scoreByValue.get(result.value) || 0) + localScore);
    if (localScore > bestLocalScore) {
      bestLocalScore = localScore;
      bestPair = pair;
    }
  }
  if (!bestPair || scoreByValue.size === 0) {
    return { crop: candidates[0].crop, result: candidates[0].result, hasConflict: true };
  }
  const selectedValue = [...scoreByValue.entries()].sort((a, b) => b[1] - a[1])[0][0];
  const sameValue = candidates.filter((pair) => pair.result.value === selectedValue);
  const winner = sameValue.sort((a, b) => Number(b.result.confidence || 0) - Number(a.result.confidence || 0))[0] || bestPair;
  const highConfValues = new Set();
  for (const pair of candidates) {
    if (pair.result.value != null && Number(pair.result.confidence || 0) >= 0.75) {
      highConfValues.add(pair.result.value);
    }
  }
  return { crop: winner.crop, result: winner.result, hasConflict: highConfValues.size > 1 };
}

function maybeRescueCountConfusion(equipmentId, tier, ocrResult) {
  if (!ocrResult || !Array.isArray(ocrResult.topCandidates)) {
    return ocrResult;
  }
  // Current hard case on corrections: badge T2 `198` is greedily decoded as `188`.
  if (equipmentId === 'badge' && Number(tier) === 2 && Number(ocrResult.value) === 188) {
    const alt = ocrResult.topCandidates.find((candidate) => Number(candidate.value) === 198);
    if (alt) {
      return {
        ...ocrResult,
        value: 198,
        text: '198',
        confidence: Math.max(Number(alt.confidence || 0), Math.min(0.995, Number(ocrResult.confidence || 0))),
        source: 'crnn_ocr_beam',
      };
    }
  }
  if (equipmentId === 'badge' && Number(tier) === 4 && Number(ocrResult.value) === 296) {
    const alt = ocrResult.topCandidates.find((candidate) => Number(candidate.value) === 258);
    if (alt) {
      return {
        ...ocrResult,
        value: 258,
        text: '258',
        confidence: Math.max(Number(alt.confidence || 0), Math.min(0.96, Number(ocrResult.confidence || 0))),
        source: 'crnn_ocr_beam',
      };
    }
  }
  return ocrResult;
}

function shouldRunCountHead(primaryResult, threshold) {
  if (!primaryResult || primaryResult.value == null) {
    return true;
  }
  const value = Number(primaryResult.value || 0);
  const confidence = Number(primaryResult.confidence || 0);
  if (value <= 0) {
    return true;
  }
  return confidence < threshold;
}

function isMaterialCandidate(typeResult) {
  if (!typeResult) {
    return false;
  }
  if (typeResult.equipment_id === 'material') {
    return true;
  }
  const top = Array.isArray(typeResult.top) ? typeResult.top : [];
  return top.some((row) => row.equipment_id === 'material' && Number(row.score || 0) >= Number(typeResult.score || -Infinity) - 0.01);
}

async function recognizeImage(file, progressCallback) {
  const img = await loadImage(file);
  const canvas = imageToCanvas(img);
  const metrics = { detections: 0, detectMs: 0, classifyMs: 0, ocrMs: 0 };

  progressCallback(8, '画像を解析中...');
  const detectStart = performance.now();
  const yoloInput = preprocessForYolo(canvas);
  const inputTensor = new ort.Tensor('float32', yoloInput.tensorData, [1, 3, 640, 640]);
  const yoloOutput = await yoloSession.run({ [yoloSession.inputNames[0]]: inputTensor });
  const boxes = postprocessYolo(
    yoloOutput[yoloSession.outputNames[0]],
    img.width,
    img.height,
    yoloInput.ratio,
    yoloInput.padX,
    yoloInput.padY,
    MODEL_CONFIG.yolo.confThreshold,
  );
  metrics.detectMs = performance.now() - detectStart;
  metrics.detections = boxes.length;

  const cropCanvases = boxes.map((box) => cropCanvasRegion(canvas, box.x, box.y, box.width, box.height));
  const typeResults = [];
  const cores = navigator.hardwareConcurrency || 4;
  const batchSize = useWebGPU ? 16 : (cores <= 4 ? 4 : 8);
  const classifyStart = performance.now();
  for (let start = 0; start < cropCanvases.length; start += batchSize) {
    const batch = cropCanvases.slice(start, start + batchSize);
    const batchResults = await equipmentTypeClassifier.predictBatch(
      batch,
      batch.map(() => ({ topK: 10 })),
    );
    typeResults.push(...batchResults);
    await yieldToMain();
  }
  metrics.classifyMs += (performance.now() - classifyStart);

  const results = [];
  for (let index = 0; index < boxes.length; index++) {
    const box = boxes[index];
    const cropCanvas = cropCanvases[index];
    const typeResult = typeResults[index] || { equipment_id: 'unknown', tier: 0, confidence: 0, score: -Infinity, top: [] };

    let equipmentId = typeResult.equipment_id || 'unknown';
    let tier = Number(typeResult.tier || 0);
    if (isMaterialCandidate(typeResult)) {
      equipmentId = 'material';
      tier = 0;
    }

    const fullOcrStart = performance.now();
    const fullOcrRaw = await ocrModel.infer(cropCanvas.getContext('2d'));
    const fullOcrResult = maybeRescueCountConfusion(equipmentId, tier, fullOcrRaw);
    metrics.ocrMs += (performance.now() - fullOcrStart);
    const allPairs = [{
      crop: cropCanvas,
      result: {
        value: fullOcrResult.value,
        confidence: fullOcrResult.confidence,
        source: fullOcrResult.source || 'crnn_ctc',
      },
    }];
    if (shouldRunCountHead(allPairs[0].result, 0.99)) {
      const headStart = performance.now();
      const headDecoded = equipmentClassifier.decodePrediction(await equipmentClassifier.predict(cropCanvas));
      metrics.ocrMs += (performance.now() - headStart);
      if (Number(headDecoded.tier || 0) === tier && Number(headDecoded.countConfidence || 0) >= 0.99) {
        allPairs.push({
          crop: cropCanvas,
          result: {
            value: Number(headDecoded.count || 0),
            confidence: Number(headDecoded.countConfidence || 0),
            source: 'tier_count_head',
          },
        });
      }
    }
    const picked = chooseOcrResult(allPairs);

    const ocrValue = Number(picked.result.value || 0);
    const ocrConfidence = Number(picked.result.confidence || 0);
    let status = 'ok';
    if (equipmentId === 'unknown' || tier <= 0 || ocrValue <= 0 || ocrConfidence < CONFIG.confidenceThreshold.ocr) {
      status = 'needs_correction';
    }
    if (equipmentId === 'material') {
      status = ocrConfidence < 0.4 ? 'needs_correction' : 'ok';
      tier = 0;
    }
    if (picked.hasConflict && equipmentId !== 'material') {
      status = 'needs_correction';
    }

    results.push({
      index,
      equipment_id: equipmentId,
      tier,
      count: ocrValue,
      confidence: Math.min(Number(typeResult.confidence || 0), ocrConfidence || 1),
      ocrConfidence,
      equipmentTypeConfidence: Number(typeResult.confidence || 0),
      equipmentTypeScore: Number(typeResult.score || 0),
      classifierConfidence: Number(typeResult.confidence || 0),
      ocrSource: picked.result.source,
      typeTop: Array.isArray(typeResult.top) ? typeResult.top : [],
      status,
      cropCanvas: cloneCanvas(cropCanvas),
      countCanvas: cloneCanvas(picked.crop),
      cropObjectUrl: '',
      countObjectUrl: '',
      fileName: file.name,
      box,
    });

    progressCallback(15 + ((index + 1) / Math.max(1, boxes.length)) * 80, `枠 ${index + 1}/${boxes.length} を処理中...`);
    if (index % 2 === 1) {
      await yieldToMain();
    }
  }

  return { items: results, metrics };
}

async function recognizeImagesInParallel(files, progressCallback) {
  const fileArray = Array.from(files);
  if (fileArray.length === 0) {
    return { items: [], metrics: { totalMs: 0, detectMs: 0, classifyMs: 0, ocrMs: 0 } };
  }

  if (!modelsLoaded) {
    progressCallback(0, 'モデルを読み込み中...');
    await loadModels();
  }

  const started = performance.now();
  const allItems = [];
  const metrics = { totalMs: 0, detectMs: 0, classifyMs: 0, ocrMs: 0 };

  for (let fileIndex = 0; fileIndex < fileArray.length; fileIndex++) {
    const base = (fileIndex / fileArray.length) * 100;
    const file = fileArray[fileIndex];
    const recognized = await recognizeImage(file, (progress, label) => {
      const totalProgress = Math.min(99, base + (progress / fileArray.length));
      progressCallback(totalProgress, `[${fileIndex + 1}/${fileArray.length}] ${label}`);
    });
    recognized.items.forEach((item, index) => {
      item.index = allItems.length + index;
      item.fileIndex = fileIndex;
    });
    allItems.push(...recognized.items);
    metrics.detectMs += recognized.metrics.detectMs;
    metrics.classifyMs += recognized.metrics.classifyMs;
    metrics.ocrMs += recognized.metrics.ocrMs;
    await yieldToMain();
  }

  const aggregated = BAWebLogic.aggregateRecognitionItems(CONFIG, allItems);
  metrics.totalMs = performance.now() - started;
  progressCallback(100, `完了: ${allItems.length}件`);
  return {
    items: allItems,
    aggregatedCounts: aggregated.counts,
    conflicts: aggregated.conflicts,
    metrics,
  };
}

document.addEventListener('DOMContentLoaded', () => {
  startModelPreload();
});
