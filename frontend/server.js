const path = require('path');
const fs = require('fs');
const express = require('express');
const cors = require('cors');
require('dotenv').config();
const { GoogleGenAI } = require('@google/genai');
const { QdrantClient } = require('@qdrant/js-client-rest');

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const COLLECTION_NAME = 'humanoid_robotics_textbook';
const EMBEDDING_MODEL = 'text-embedding-004'; 

// --- Qdrant Cloud Client ---
const qdrantClient = new QdrantClient({
    url: process.env.QDRANT_URL,
    apiKey: process.env.QDRANT_API_KEY,
});

async function ingestData() {
    console.log('--- Connecting to Qdrant Cloud ---');
    try {
        await qdrantClient.getCollections();
        console.log("Qdrant Cloud Connection: SUCCESSFUL");
        
        // Data file path check
        const filePath = path.join(__dirname, 'data', 'chapter_text.txt');
        if (fs.existsSync(filePath)) {
            console.log("Found chapter_text.txt, starting ingestion...");
            // (Baaki ingestion logic yahan background mein chal sakti hai)
        }
    } catch (error) {
        console.error("Qdrant Connection FAILED. Check your .env file!");
        console.error("Error Detail:", error.message);
    }
}

const app = express();
app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
  res.status(200).json({ status: 'OK', message: 'Cloud Server is running' });
});

app.post('/api/chat', async (req, res) => {
    // Chat logic simplified for testing
    res.json({ response: "Backend is connected to Cloud!" });
});

ingestData();

app.listen(3003, () => {
  console.log(`Server is running on port 3003`);
  console.log(`Health check: http://localhost:3003/health`);
});