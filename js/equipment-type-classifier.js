class EquipmentTypeClassifier {
  constructor(
    modelPath = 'models/equipment_type_classifier.onnx',
    metadataPath = 'models/equipment_type_classifier.json',
  ) {
    this.modelPath = modelPath;
    this.metadataPath = metadataPath;
    this.session = null;
    this.labels = [];
    this.inputSize = 160;
    this.initialized = false;
    this.mean = [0.485, 0.456, 0.406];
    this.std = [0.229, 0.224, 0.225];
  }

  async initialize(executionProviders = ['wasm']) {
    if (!window.ort) {
      throw new Error('ONNX Runtime Web not loaded');
    }
    const response = await fetch(this.metadataPath);
    if (!response.ok) {
      throw new Error(`Failed to load classifier metadata: ${response.status}`);
    }
    const metadata = await response.json();
    this.labels = Array.isArray(metadata.labels) ? metadata.labels : [];
    this.inputSize = Number(metadata.input_size || 160);
    if (!this.labels.length) {
      throw new Error('equipment classifier labels are empty');
    }
    this.session = await window.ort.InferenceSession.create(this.modelPath, {
      executionProviders,
      graphOptimizationLevel: 'all',
    });
    this.initialized = true;
  }

  _softmax(logits) {
    let max = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > max) {
        max = logits[i];
      }
    }
    const out = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      const value = Math.exp(logits[i] - max);
      out[i] = value;
      sum += value;
    }
    const denom = sum > 0 ? sum : 1;
    for (let i = 0; i < out.length; i++) {
      out[i] /= denom;
    }
    return out;
  }

  _prepareInput(imageData) {
    let canvas;
    if (imageData instanceof HTMLCanvasElement) {
      canvas = imageData;
    } else if (imageData instanceof ImageData) {
      canvas = document.createElement('canvas');
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      canvas.getContext('2d').putImageData(imageData, 0, 0);
    } else {
      throw new Error('Unsupported image input');
    }

    const resized = document.createElement('canvas');
    resized.width = this.inputSize;
    resized.height = this.inputSize;
    const ctx = resized.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'medium';
    ctx.drawImage(canvas, 0, 0, this.inputSize, this.inputSize);

    const data = ctx.getImageData(0, 0, this.inputSize, this.inputSize).data;
    const out = new Float32Array(3 * this.inputSize * this.inputSize);
    const plane = this.inputSize * this.inputSize;
    for (let i = 0; i < plane; i++) {
      const r = data[i * 4] / 255.0;
      const g = data[i * 4 + 1] / 255.0;
      const b = data[i * 4 + 2] / 255.0;
      out[i] = (r - this.mean[0]) / this.std[0];
      out[plane + i] = (g - this.mean[1]) / this.std[1];
      out[plane * 2 + i] = (b - this.mean[2]) / this.std[2];
    }
    return out;
  }

  _decodeLogits(logits, topK = 8) {
    const probs = this._softmax(logits);
    const ranked = [];
    for (let i = 0; i < probs.length; i++) {
      ranked.push({ index: i, confidence: probs[i] });
    }
    ranked.sort((a, b) => b.confidence - a.confidence);
    const top = ranked.slice(0, topK).map((row) => {
      const label = String(this.labels[row.index] || 'unknown/tier_0');
      const match = /^([^/]+)\/tier_(\d+)$/.exec(label);
      return {
        label,
        equipment_id: match ? match[1] : 'unknown',
        tier: match ? Number(match[2]) : 0,
        confidence: Number(row.confidence),
        score: Number(row.confidence),
      };
    });
    const best = top[0] || {
      label: 'unknown/tier_0',
      equipment_id: 'unknown',
      tier: 0,
      confidence: 0,
      score: 0,
    };
    return {
      equipment_id: best.equipment_id,
      tier: best.tier,
      confidence: best.confidence,
      score: best.score,
      top,
    };
  }

  async predict(imageData, options = {}) {
    const results = await this.predictBatch([imageData], [options]);
    return results[0];
  }

  async predictBatch(imagesData, optionsList = []) {
    if (!this.initialized || !this.session) {
      throw new Error('Equipment type classifier not initialized');
    }
    if (!imagesData.length) {
      return [];
    }
    const plane = 3 * this.inputSize * this.inputSize;
    const batchInput = new Float32Array(imagesData.length * plane);
    for (let index = 0; index < imagesData.length; index++) {
      batchInput.set(this._prepareInput(imagesData[index]), index * plane);
    }
    const tensor = new window.ort.Tensor('float32', batchInput, [imagesData.length, 3, this.inputSize, this.inputSize]);
    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    const outputs = await this.session.run({ [inputName]: tensor });
    const raw = outputs[outputName].data;
    const dims = outputs[outputName].dims;
    const batch = dims[0];
    const classCount = dims[dims.length - 1];
    const results = [];
    for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
      const start = batchIndex * classCount;
      const logits = raw.slice(start, start + classCount);
      const topK = Number((optionsList[batchIndex] || {}).topK || 8);
      results.push(this._decodeLogits(logits, topK));
    }
    return results;
  }
}
