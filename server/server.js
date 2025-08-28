// server.js

const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();

// ✅ Enable CORS for all origins (for development)
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  next();
});

// Handle preflight requests
app.options('/analyze', (req, res) => {
  res.sendStatus(200);
});

// Parse JSON
app.use(express.json());

// 🔐 Your API key
const API_KEY = "AIzaSyAEqZ0C0c223ztnrkLpThMzYkr_en4wwEE";
const genAI = new GoogleGenerativeAI(API_KEY);

// ✅ Serve static files (your HTML page)
app.use(express.static('public')); // This serves index.html from /public

// POST /analyze — Analyze email
app.post('/analyze', async (req, res) => {
  const { text } = req.body;

  if (!text || typeof text !== 'string') {
    return res.status(400).json({ error: 'Email text is required' });
  }

  try {
    // ✅ Fix: Use correct model name
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' }); // or 'gemini-pro'

    const prompt = `
      Analyze the following email and classify it as "spam" or "ham".
      Respond with only one word: "spam" or "ham". No explanation.

      Email:
      "${text.trim()}"
    `;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const output = response.text().trim().toLowerCase();

    let isSpam;
    if (output.includes('spam')) {
      isSpam = true;
    } else if (output.includes('ham')) {
      isSpam = false;
    } else {
      console.warn('Unclear output:', output);
      isSpam = Math.random() > 0.5;
    }

    res.json({ isSpam });
  } catch (error) {
    console.error('Gemini Error:', error);
    res.status(500).json({ error: 'Failed to analyze email' });
  }
});

// Start server
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
  console.log(`👉 Open http://localhost:${PORT} in your browser`);
});