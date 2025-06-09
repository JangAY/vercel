import * as tf from '@tensorflow/tfjs';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

let model;
let tokenizer;

const labels = ['neutral', 'sadness', 'anger', 'fear', 'suicidal'];
const MAX_LEN = 100;

// Load model & tokenizer
async function loadResources() {
  if (!model) {
    model = await tf.loadLayersModel('/public/model.json');
    console.log('✅ Model loaded');
  }
  if (!tokenizer) {
    const tokenizerPath = path.join(__dirname, '../public/tokenizer_health.json');
    const tokenizerRaw = await fs.readFile(tokenizerPath, 'utf-8');
    tokenizer = JSON.parse(tokenizerRaw).word_index || JSON.parse(tokenizerRaw);
    console.log('✅ Tokenizer loaded');
  }
}

// Preprocessing
function preprocess(text) {
  const tokens = text.toLowerCase().split(/\s+/).map(w => tokenizer[w] || 0);
  const padded = new Array(MAX_LEN).fill(0);
  for (let i = 0; i < Math.min(tokens.length, MAX_LEN); i++) {
    padded[i] = tokens[i];
  }
  return tf.tensor2d([padded]);
}

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Only POST allowed' });

  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: 'Missing text' });

    await loadResources();

    const input = preprocess(text);
    const prediction = model.predict(input);
    const result = await prediction.data();
    const scores = Array.from(result);
    const maxIndex = scores.indexOf(Math.max(...scores));

    res.status(200).json({
      label: labels[maxIndex],
      scores: Object.fromEntries(labels.map((l, i) => [l, scores[i]]))
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Prediction failed', details: err.message });
  }
}
