import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';

// Load tokenizer dari file JSON (dimuat satu kali saat cold start)
const tokenizerPath = path.join(process.cwd(), 'public', 'tokenizer_health.json');
const tokenizerData = JSON.parse(fs.readFileSync(tokenizerPath, 'utf8'));

// Load model dari direktori publik (satu kali)
let model;
const modelLoadPath = `file://${path.join(process.cwd(), 'public', 'model.json')}`;

async function loadModelOnce() {
  if (!model) {
    model = await tf.loadLayersModel(modelLoadPath);
    console.log('✅ Model loaded');
  }
}
await loadModelOnce();

// Fungsi untuk mengubah teks menjadi tensor input
function preprocessText(text) {
  const maxLen = 100; // Sama dengan panjang maksimal saat training
  const wordIndex = tokenizerData.word_index || tokenizerData;

  const tokens = text.toLowerCase().split(/\s+/).map(w => wordIndex[w] || 0);
  const padded = new Array(maxLen).fill(0);
  for (let i = 0; i < Math.min(tokens.length, maxLen); i++) {
    padded[i] = tokens[i];
  }
  return tf.tensor2d([padded]);
}

// Daftar label emosi (disesuaikan dengan output model)
const labels = ['neutral', 'sadness', 'anger', 'fear', 'suicidal'];

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const { text } = req.body;
    if (!text || typeof text !== 'string') {
      return res.status(400).json({ error: 'Text is required' });
    }

    const inputTensor = preprocessText(text);
    const prediction = model.predict(inputTensor);
    const result = await prediction.data();
    const scores = Array.from(result);
    const maxIndex = scores.indexOf(Math.max(...scores));
    const predictedLabel = labels[maxIndex];

    res.status(200).json({
      label: predictedLabel,
      scores: Object.fromEntries(labels.map((label, i) => [label, scores[i]]))
    });

  } catch (err) {
    console.error('Prediction error:', err);
    res.status(500).json({ error: 'Internal error', details: err.message });
  }
}
