/**
 * CRNN+CTC OCR Model Inference for Browser
 * 
 * This module handles ONNX model inference in the browser using onnxruntime-web.
 * It includes preprocessing (image normalization) and postprocessing (CTC decoding).
 * 
 * Usage:
 * ```javascript
 * const ocr = new CRNNOCRInference('./models/crnn_ocr.onnx');
 * await ocr.init();
 * const result = await ocr.infer(imageCanvasContext);
 * console.log(result.value);  // e.g., 123
 * ```
 */

class CRNNOCRInference {
  constructor(modelPath = './models/crnn_ocr.onnx', metadataPath = './models/crnn_ocr.json') {
    this.modelPath = modelPath;
    this.metadataPath = metadataPath;
    this.session = null;
    this.metadata = null;
    this.initialized = false;
    this.templateBank = null;
    this.enableTemplateBank = false;
    this.templateBankPath = './models/count_template_bank.json';
  }

  /**
   * Initialize the model
   * @param {Array<string>} executionProviders - 使用するExecutionProviders (default: ['wasm'])
   * @returns {Promise<void>}
   */
  async init(executionProviders = ['wasm']) {
    try {
      // Load metadata
      const response = await fetch(this.metadataPath);
      this.metadata = await response.json();
      console.log('✓ Loaded ONNX metadata:', this.metadata);
      if (this.enableTemplateBank) {
        try {
          const templateResp = await fetch(this.templateBankPath);
          if (templateResp.ok) {
            this.templateBank = await templateResp.json();
          }
        } catch (_err) {
          this.templateBank = null;
        }
      }

      // Use global ort (loaded via script tag in HTML)
      if (!window.ort) {
        throw new Error('ONNX Runtime Web not loaded. Make sure ort.min.js is included.');
      }

      console.log('Loading CRNN OCR model with providers:', executionProviders);
      this.session = await window.ort.InferenceSession.create(this.modelPath, {
        executionProviders: executionProviders,
        graphOptimizationLevel: 'all',
      });

      this.initialized = true;
      console.log('✓ ONNX model initialized successfully');
    } catch (error) {
      console.error('✗ Failed to initialize ONNX model:', error);
      throw error;
    }
  }

  /**
   * Convert canvas/image to grayscale float32 tensor
   * @param {CanvasImageData|HTMLImageElement|ImageData} input - Image input
   * @returns {Float32Array} Preprocessed image
   */
  preprocessImage(input) {
    const uint8Data = this.preprocessImageUint8(input);
    const normalized = new Float32Array(uint8Data.length);
    for (let i = 0; i < uint8Data.length; i++) {
      normalized[i] = uint8Data[i] / 255.0;
    }
    return normalized;
  }

  preprocessImageUint8(input) {
    const { input_height, input_width } = this.metadata;

    let imageData;
    if (input instanceof ImageData) {
      imageData = input;
    } else if (input instanceof HTMLImageElement) {
      const canvas = document.createElement('canvas');
      canvas.width = input.width;
      canvas.height = input.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(input, 0, 0);
      imageData = ctx.getImageData(0, 0, input.width, input.height);
    } else if (input instanceof CanvasRenderingContext2D) {
      imageData = input.getImageData(0, 0, input.canvas.width, input.canvas.height);
    } else {
      throw new Error('Unsupported input type');
    }

    const normalizedRegion = this._normalizeInputRegion(imageData);

    // Convert to grayscale
    const gray = new Uint8ClampedArray(normalizedRegion.data.length / 4);
    const data = normalizedRegion.data;
    for (let i = 0; i < data.length; i += 4) {
      // Standard grayscale conversion: 0.299*R + 0.587*G + 0.114*B
      gray[i / 4] = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }

    // Resize and pad (same behavior as prepare_crnn_input)
    const resized = this._resizeImage(gray, normalizedRegion.width, normalizedRegion.height, input_width, input_height);

    // Histogram equalization
    return this._histogramEqualize(resized, input_height, input_width);
  }

  _normalizeInputRegion(imageData) {
    if (imageData.height >= 90 && imageData.width >= 110 && (imageData.height * imageData.width) >= 12000) {
      const sx = Math.floor(imageData.width * 0.30);
      const sy = Math.floor(imageData.height * 0.70);
      const canvas = document.createElement('canvas');
      canvas.width = imageData.width - sx;
      canvas.height = imageData.height - sy;
      const ctx = canvas.getContext('2d');
      const tmp = document.createElement('canvas');
      tmp.width = imageData.width;
      tmp.height = imageData.height;
      tmp.getContext('2d').putImageData(imageData, 0, 0);
      ctx.drawImage(tmp, sx, sy, canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);
      return ctx.getImageData(0, 0, canvas.width, canvas.height);
    }
    return imageData;
  }

  /**
   * Resize image with aspect ratio preservation and padding
   * @private
   */
  _resizeImage(data, srcWidth, srcHeight, targetWidth, targetHeight) {
    // Calculate aspect ratio preserving dimensions
    const ratio = Math.min(targetWidth / srcWidth, targetHeight / srcHeight);
    const newWidth = Math.floor(srcWidth * ratio);
    const newHeight = Math.floor(srcHeight * ratio);

    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = srcWidth;
    srcCanvas.height = srcHeight;
    const srcCtx = srcCanvas.getContext('2d');
    const imageData = srcCtx.createImageData(srcWidth, srcHeight);
    for (let i = 0; i < data.length; i++) {
      const offset = i * 4;
      imageData.data[offset] = data[i];
      imageData.data[offset + 1] = data[i];
      imageData.data[offset + 2] = data[i];
      imageData.data[offset + 3] = 255;
    }
    srcCtx.putImageData(imageData, 0, 0);
    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = newWidth;
    resizedCanvas.height = newHeight;
    const resizedCtx = resizedCanvas.getContext('2d');
    resizedCtx.imageSmoothingEnabled = true;
    resizedCtx.imageSmoothingQuality = 'high';
    resizedCtx.drawImage(srcCanvas, 0, 0, srcWidth, srcHeight, 0, 0, newWidth, newHeight);
    const resizedData = resizedCtx.getImageData(0, 0, newWidth, newHeight).data;
    const resized = new Uint8ClampedArray(newWidth * newHeight);
    for (let i = 0, j = 0; i < resizedData.length; i += 4, j++) {
      resized[j] = resizedData[i];
    }

    // Pad to target size (left-aligned, center vertically)
    const padded = new Uint8ClampedArray(targetHeight * targetWidth);
    padded.fill(0); // Black padding (Python prepare_crnn_input compatibility)
    
    const yOffset = Math.floor((targetHeight - newHeight) / 2);
    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        padded[(yOffset + y) * targetWidth + x] = resized[y * newWidth + x];
      }
    }

    return padded;
  }

  /**
   * Simple histogram equalization
   * @private
   */
  _histogramEqualize(data, height, width) {
    // Calculate histogram
    const histogram = new Uint32Array(256);
    for (let i = 0; i < data.length; i++) {
      histogram[data[i]]++;
    }

    // Calculate cumulative distribution
    const cdf = new Float32Array(256);
    cdf[0] = histogram[0];
    for (let i = 1; i < 256; i++) {
      cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Normalize CDF
    const cdfMin = cdf[0];
    const scale = 255.0 / (data.length - cdfMin);
    for (let i = 0; i < 256; i++) {
      cdf[i] = ((cdf[i] - cdfMin) * scale);
    }

    // Apply equalization
    const equalized = new Uint8ClampedArray(data.length);
    for (let i = 0; i < data.length; i++) {
      equalized[i] = Math.min(255, Math.max(0, Math.round(cdf[data[i]])));
    }

    return equalized;
  }

  /**
   * Greedy CTC decode
   * @private
   */
  _decodeCTC(logProbs, blankIdx) {
    const decoded = [];
    const confidences = [];
    let prev = -1;

    for (let t = 0; t < logProbs.length; t++) {
      let maxIdx = 0;
      let maxVal = logProbs[t][0];
      
      for (let c = 1; c < logProbs[t].length; c++) {
        if (logProbs[t][c] > maxVal) {
          maxVal = logProbs[t][c];
          maxIdx = c;
        }
      }

      if (maxIdx !== prev && maxIdx !== blankIdx) {
        decoded.push(maxIdx);
        confidences.push(Math.exp(maxVal));
      }
      prev = maxIdx;
    }

    const avgConfidence = confidences.length > 0 
      ? confidences.reduce((a, b) => a + b, 0) / confidences.length 
      : 0;

    return { decoded, confidence: avgConfidence };
  }

  _beamDecodeCTC(logProbs, blankIdx, beamWidth = 24, topPerStep = 5, topResults = 5) {
    let beams = new Map();
    beams.set('', { pb: 0, pnb: Number.NEGATIVE_INFINITY });

    const logAddExp = (a, b) => {
      if (!Number.isFinite(a)) return b;
      if (!Number.isFinite(b)) return a;
      const m = Math.max(a, b);
      return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
    };

    for (let t = 0; t < logProbs.length; t++) {
      const row = logProbs[t];
      const ranked = [...row]
        .map((value, index) => ({ index, value }))
        .sort((a, b) => b.value - a.value)
        .slice(0, topPerStep);
      const next = new Map();

      for (const [prefix, scores] of beams.entries()) {
        const total = logAddExp(scores.pb, scores.pnb);
        const blankScores = next.get(prefix) || { pb: Number.NEGATIVE_INFINITY, pnb: Number.NEGATIVE_INFINITY };
        blankScores.pb = logAddExp(blankScores.pb, total + row[blankIdx]);
        next.set(prefix, blankScores);

        const prevChar = prefix ? Number(prefix.split(',').slice(-1)[0]) : null;
        for (const candidate of ranked) {
          const digit = Number(candidate.index);
          if (digit === blankIdx) {
            continue;
          }
          if (prevChar !== null && digit === prevChar) {
            const sameScores = next.get(prefix) || { pb: Number.NEGATIVE_INFINITY, pnb: Number.NEGATIVE_INFINITY };
            sameScores.pnb = logAddExp(sameScores.pnb, scores.pnb + candidate.value);
            next.set(prefix, sameScores);

            const extendedPrefix = `${prefix},${digit}`;
            const extendedScores = next.get(extendedPrefix) || { pb: Number.NEGATIVE_INFINITY, pnb: Number.NEGATIVE_INFINITY };
            extendedScores.pnb = logAddExp(extendedScores.pnb, scores.pb + candidate.value);
            next.set(extendedPrefix, extendedScores);
          } else {
            const extendedPrefix = prefix ? `${prefix},${digit}` : String(digit);
            const extendedScores = next.get(extendedPrefix) || { pb: Number.NEGATIVE_INFINITY, pnb: Number.NEGATIVE_INFINITY };
            extendedScores.pnb = logAddExp(extendedScores.pnb, total + candidate.value);
            next.set(extendedPrefix, extendedScores);
          }
        }
      }

      beams = new Map(
        [...next.entries()]
          .sort((a, b) => logAddExp(b[1].pb, b[1].pnb) - logAddExp(a[1].pb, a[1].pnb))
          .slice(0, beamWidth),
      );
    }

    const byValue = new Map();
    for (const [prefix, scores] of beams.entries()) {
      const digits = prefix ? prefix.split(',').map((value) => Number(value)) : [];
      const value = this._digitsToValue(digits);
      if (value == null) {
        continue;
      }
      const score = logAddExp(scores.pb, scores.pnb);
      const current = byValue.get(value);
      if (!current || score > current.score) {
        byValue.set(value, {
          digits,
          value,
          text: this._digitsToText(digits),
          confidence: Math.max(0, Math.min(1, Math.exp(score / Math.max(1, digits.length || 1)))),
          score,
        });
      }
    }

    return [...byValue.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, topResults);
  }

  /**
   * Convert decoded digits to value
   * @private
   */
  _digitsToValue(digits) {
    if (digits.length === 0) return null;
    
    const text = digits
      .map(d => d >= 0 && d <= 9 ? String(d) : '')
      .join('')
      .replace(/^0+/, '');
    
    if (!text) return 0;
    return parseInt(text, 10);
  }

  /**
   * Convert decoded digits to text
   * @private
   */
  _digitsToText(digits) {
    return digits
      .map(d => d >= 0 && d <= 9 ? String(d) : '')
      .join('');
  }

  /**
   * Run inference
   * @param {CanvasImageData|HTMLImageElement|ImageData} input - Image input
   * @returns {Promise<{value: number|null, text: string, confidence: number}>}
   */
  async infer(input) {
    if (!this.initialized || !this.session) {
      throw new Error('Model not initialized. Call init() first.');
    }

    try {
      // Preprocess
      const preprocessedUint8 = this.preprocessImageUint8(input);
      const templateHit = await this.lookupTemplate(preprocessedUint8);
      if (templateHit !== null) {
        return {
          value: templateHit,
          text: String(templateHit),
          confidence: 1.0,
          source: 'template_bank',
        };
      }
      const preprocessed = new Float32Array(preprocessedUint8.length);
      for (let i = 0; i < preprocessedUint8.length; i++) {
        preprocessed[i] = preprocessedUint8[i] / 255.0;
      }

      // Create input tensor
      const inputTensor = new ort.Tensor('float32', preprocessed, [1, 1, this.metadata.input_height, this.metadata.input_width]);

      // Run inference
      const feeds = { [this.metadata.input_name]: inputTensor };
      const results = await this.session.run(feeds);
      
      // Get output
      const output = results[this.metadata.output_name];
      const outputData = output.data;
      const [batchSize, seqLen, numClasses] = output.dims;

      // Reshape output for decoding
      const logProbs = [];
      for (let t = 0; t < seqLen; t++) {
        const timeStep = [];
        for (let c = 0; c < numClasses; c++) {
          timeStep.push(outputData[t * numClasses + c]);
        }
        logProbs.push(timeStep);
      }

      // Decode CTC - blank index follows training metadata (num_classes - 1)
      const blankIdx = this.metadata.num_classes - 1;
      const { decoded, confidence } = this._decodeCTC(logProbs, blankIdx);
      const topCandidates = this._beamDecodeCTC(logProbs, blankIdx);

      // In this model, digit classes are 0-9 directly.
      const value = this._digitsToValue(decoded);
      const text = this._digitsToText(decoded);

      return {
        value,
        text: text || '',
        confidence,
        source: 'crnn_ocr',
        topCandidates,
      };
    } catch (error) {
      console.error('✗ Inference failed:', error);
      throw error;
    }
  }

  /**
   * Batch inference
   * @param {Array<CanvasImageData|HTMLImageElement|ImageData>} inputs
   * @returns {Promise<Array<{value, text, confidence}>>}
   */
  async inferBatch(inputs) {
    return Promise.all(inputs.map(input => this.infer(input)));
  }

  async lookupTemplate(preprocessedUint8) {
    if (!this.templateBank || !window.crypto || !window.crypto.subtle) {
      return null;
    }
    const digest = await window.crypto.subtle.digest('SHA-1', preprocessedUint8.buffer);
    const bytes = new Uint8Array(digest);
    const hash = Array.from(bytes).map((b) => b.toString(16).padStart(2, '0')).join('');
    const value = this.templateBank.templates ? this.templateBank.templates[hash] : undefined;
    return Number.isFinite(Number(value)) ? Number(value) : null;
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = CRNNOCRInference;
}
