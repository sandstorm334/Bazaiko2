/**
 * Equipment Tier/Count Classifier ONNX Model Loader
 * Browser-compatible JavaScript for loading and running the ONNX classifier model
 */

class EquipmentClassifier {
    /**
     * Initialize the classifier
     * @param {string} modelPath - Path to classifier.onnx
     * @param {string} configPath - Path to classifier_config.json
     */
    constructor(modelPath, configPath) {
        this.modelPath = modelPath;
        this.configPath = configPath;
        this.session = null;
        this.config = null;
        this.ready = false;
    }

    /**
     * Load model configuration
     */
    async loadConfig() {
        const response = await fetch(this.configPath);
        this.config = await response.json();
        console.log("Classifier config loaded:", this.config);
    }

    /**
     * Initialize ONNX Runtime and load model
     * Requires onnxruntime-web library loaded
     * @param {Array<string>} executionProviders - 使用するExecutionProviders (default: ['wasm'])
     */
    async initialize(executionProviders = ['wasm']) {
        if (!window.ort) {
            throw new Error("ONNX Runtime Web not found. Include onnxruntime-web library.");
        }

        // Load config first
        if (!this.config) {
            await this.loadConfig();
        }

        // Load model with WebGPU support
        console.log("Loading ONNX model from:", this.modelPath, "with providers:", executionProviders);
        this.session = await window.ort.InferenceSession.create(this.modelPath, {
            executionProviders: executionProviders,
            graphOptimizationLevel: 'all',
        });

        this.ready = true;
        console.log("Classifier model loaded and ready");
    }

    /**
     * Prepare image for inference (matches Python preprocessing)
     * @param {HTMLCanvasElement|ImageData} imageData - Image to process
     * @returns {Float32Array} Normalized tensor ready for model
     */
    prepareImage(imageData) {
        if (!this.config) {
            throw new Error("Config not loaded");
        }

        const inputSize = this.config.input_size; // 128
        let canvas, ctx;

        // Convert to canvas if needed
        if (imageData instanceof HTMLCanvasElement) {
            canvas = imageData;
        } else if (imageData instanceof ImageData) {
            canvas = document.createElement("canvas");
            canvas.width = imageData.width;
            canvas.height = imageData.height;
            ctx = canvas.getContext("2d");
            ctx.putImageData(imageData, 0, 0);
        } else if (imageData instanceof HTMLImageElement) {
            canvas = document.createElement("canvas");
            canvas.width = imageData.width;
            canvas.height = imageData.height;
            ctx = canvas.getContext("2d");
            ctx.drawImage(imageData, 0, 0);
        } else {
            throw new Error("Invalid image input");
        }

        // Get canvas context
        ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;

        // Convert to grayscale
        const imgData = ctx.getImageData(0, 0, w, h);
        const data = imgData.data;
        const gray = new Uint8Array(w * h);
        for (let i = 0, j = 0; i < data.length; i += 4, j++) {
            // Standard grayscale conversion: 0.299*R + 0.587*G + 0.114*B
            gray[j] = Math.round(
                0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
            );
        }

        // Resize full image
        const full = this.resizeGrayscale(gray, w, h, inputSize);

        // Tier crop: lower-left area (58% to bottom, 0% to 50%)
        const tierStartY = Math.floor(h * 0.58);
        const tierEndX = Math.floor(w * 0.5);
        const tierCrop = this.cropAndResize(
            gray,
            w,
            h,
            0,
            tierStartY,
            tierEndX,
            h,
            inputSize
        );

        // Count crop: lower-right area (68% to bottom, 30% to right)
        const countStartY = Math.floor(h * 0.68);
        const countStartX = Math.floor(w * 0.3);
        const countCrop = this.cropAndResize(
            gray,
            w,
            h,
            countStartX,
            countStartY,
            w,
            h,
            inputSize
        );

        // Stack channels: [full, tier_crop, count_crop] and normalize to [0, 1]
        const stacked = new Float32Array(3 * inputSize * inputSize);
        let idx = 0;

        // Full image
        for (let i = 0; i < full.length; i++) {
            stacked[idx++] = full[i] / 255.0;
        }

        // Tier crop
        for (let i = 0; i < tierCrop.length; i++) {
            stacked[idx++] = tierCrop[i] / 255.0;
        }

        // Count crop
        for (let i = 0; i < countCrop.length; i++) {
            stacked[idx++] = countCrop[i] / 255.0;
        }

        return stacked;
    }

    /**
     * Resize grayscale image using simple nearest-neighbor (can use better algo)
     */
    resizeGrayscale(gray, srcW, srcH, dstSize) {
        const dst = new Uint8Array(dstSize * dstSize);
        const xRatio = srcW / dstSize;
        const yRatio = srcH / dstSize;

        for (let y = 0; y < dstSize; y++) {
            for (let x = 0; x < dstSize; x++) {
                const srcX = Math.floor(x * xRatio);
                const srcY = Math.floor(y * yRatio);
                dst[y * dstSize + x] = gray[srcY * srcW + srcX];
            }
        }

        return dst;
    }

    /**
     * Crop and resize region
     */
    cropAndResize(gray, srcW, srcH, x1, y1, x2, y2, dstSize) {
        const cropW = Math.max(1, x2 - x1);
        const cropH = Math.max(1, y2 - y1);
        const dst = new Uint8Array(dstSize * dstSize);

        const xRatio = cropW / dstSize;
        const yRatio = cropH / dstSize;

        for (let y = 0; y < dstSize; y++) {
            for (let x = 0; x < dstSize; x++) {
                const srcX = Math.floor(x1 + x * xRatio);
                const srcY = Math.floor(y1 + y * yRatio);
                const clampedX = Math.max(0, Math.min(srcW - 1, srcX));
                const clampedY = Math.max(0, Math.min(srcH - 1, srcY));
                dst[y * dstSize + x] = gray[clampedY * srcW + clampedX];
            }
        }

        return dst;
    }

    /**
     * Run inference
     * @param {Tensor} input - Input tensor (batch_size, 3, 128, 128)
     * @returns {Promise<{tierLogits, lengthLogits, countLogits}>}
     */
    async predict(imageData) {
        if (!this.ready || !this.session) {
            throw new Error("Classifier not initialized. Call initialize() first.");
        }

        // Prepare input
        const preparedInput = this.prepareImage(imageData);

        // Add batch dimension: (1, 3, 128, 128)
        const input = new window.ort.Tensor("float32", preparedInput, [
            1,
            this.config.input_channels,
            this.config.input_size,
            this.config.input_size,
        ]);

        // Run inference
        const results = await this.session.run({ input });

        // Extract outputs
        const tierLogits = results.tier_logits.data;
        const lengthLogits = results.length_logits.data;
        const countLogits = results.count_logits.data;

        return {
            tierLogits: new Float32Array(tierLogits),
            lengthLogits: new Float32Array(lengthLogits),
            countLogits: new Float32Array(countLogits),
        };
    }

    /**
     * Decode prediction to human-readable format
     * @param {Object} predictions - Output from predict()
     * @returns {Object} Decoded prediction
     */
    decodePrediction(predictions) {
        const config = this.config;

        // Tier prediction
        const tierLogits = predictions.tierLogits;
        const tierIdx = this.argMax(tierLogits);
        const tierProbs = this.softmax(tierLogits);
        const tierConf = tierProbs[tierIdx];

        // Length prediction
        const lengthLogits = predictions.lengthLogits;
        const lengthIdx = this.argMax(lengthLogits);
        const lengthValue = lengthIdx + 1; // Offset by 1 for actual digit count
        const lengthProbs = this.softmax(lengthLogits);
        const lengthConf = lengthProbs[lengthIdx];

        // Count prediction
        const countLogits = predictions.countLogits;
        const countDigitPreds = [];
        const digitConfs = [];

        for (let pos = 0; pos < config.max_count_digits; pos++) {
            const posLogits = [];
            for (let d = 0; d < config.digit_classes; d++) {
                posLogits.push(
                    countLogits[pos * config.digit_classes + d]
                );
            }
            const digitIdx = this.argMax(posLogits);
            const digitProbs = this.softmax(posLogits);
            countDigitPreds.push(digitIdx);

            // Only consider confidence for actual digits (up to lengthValue)
            if (pos < lengthValue) {
                digitConfs.push(digitProbs[digitIdx]);
            }
        }

        // Decode count value
        const countStr = countDigitPreds
            .slice(0, lengthValue)
            .join("")
            .replace(/^0+/, ""); // Remove leading zeros
        const countValue = countStr ? parseInt(countStr) : 0;
        const countConf =
            digitConfs.length > 0
                ? digitConfs.reduce((a, b) => a + b) / digitConfs.length
                : 0;

        // Overall confidence
        const confidence = Math.min(tierConf, countConf);

        return {
            tier: tierIdx,
            tierConfidence: tierConf,
            count: countValue,
            countConfidence: countConf,
            lengthDigits: lengthValue,
            lengthConfidence: lengthConf,
            confidence: confidence,
            allDigits: countDigitPreds,
        };
    }

    /**
     * Softmax function
     */
    softmax(logits) {
        // Subtract max for numerical stability
        const max = Math.max(...logits);
        const exps = Array.from(logits).map((x) => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b);
        return exps.map((x) => x / sum);
    }

    /**
     * ArgMax function
     */
    argMax(arr) {
        let maxIdx = 0;
        let maxVal = arr[0];
        for (let i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}

// Export for module systems
if (typeof module !== "undefined" && module.exports) {
    module.exports = EquipmentClassifier;
}
