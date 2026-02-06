// ingest.js
const fs = require('fs/promises');
const path = require('path');
const { GoogleGenAI } = require('@google/genai');
const QdrantPackage = require('qdrant-client');
const QdrantClient = QdrantPackage.QdrantClient;
require('dotenv').config();

// Configuration
const CHAPTERS_DIR = path.join(__dirname, 'docs');
const COLLECTION_NAME = 'humanoid_robotics';
const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6336'; // Naya port 6336!
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const EMBEDDING_MODEL = 'embedding-001'; // Gemini's embedding model

if (!GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not set in the .env file");
}

const ai = new GoogleGenAI(GEMINI_API_KEY);
const qdrant = new QdrantClient({ url: QDRANT_URL });

/**
 * Text ko chote hisson (chunks) mein todna.
 * (Production mein behtar splitting logic ki zaroorat hoti hai)
 */
function chunkText(text, maxWords = 150) {
    const words = text.split(/\s+/);
    const chunks = [];
    let currentChunk = [];

    for (const word of words) {
        currentChunk.push(word);
        if (currentChunk.length >= maxWords) {
            chunks.push(currentChunk.join(' '));
            currentChunk = [];
        }
    }
    if (currentChunk.length > 0) {
        chunks.push(currentChunk.join(' '));
    }
    return chunks;
}

/**
 * Embedding create karna (vector banana)
 */
async function createEmbedding(text) {
    try {
        const response = await ai.models.embedContent({
            model: EMBEDDING_MODEL,
            content: text,
        });
        return response.embedding.values;
    } catch (error) {
        console.error("Error creating embedding with Gemini:", error.message);
        return null;
    }
}

/**
 * Qdrant mein collection banana aur data daalna
 */
async function ingestData() {
    try {
        console.log("Starting data ingestion with Gemini and Qdrant...");

        // 1. Collection check aur create karna
        const collections = await qdrant.getCollections();
        if (!collections.collections.some(c => c.name === COLLECTION_NAME)) {
            console.log(`Creating collection: ${COLLECTION_NAME}`);
            await qdrant.createCollection(COLLECTION_NAME, {
                vectors_config: {
                    // Ye 'embedding-001' ki default dimension hai
                    size: 768,
                    distance: 'Cosine',
                },
            });
            console.log("Collection created.");
        } else {
            console.log(`Collection ${COLLECTION_NAME} already exists. Skipping creation.`);
        }

        // 2. Chapters ko read aur process karna
        const files = await fs.readdir(CHAPTERS_DIR);
        const mdFiles = files.filter(f => f.endsWith('.md'));

        let pointId = 0;
        const points = [];

        for (const file of mdFiles) {
            console.log(`\nProcessing file: ${file}`);
            const filePath = path.join(CHAPTERS_DIR, file);
            const content = await fs.readFile(filePath, 'utf-8');
            const chunks = chunkText(content);

            for (const [index, chunk] of chunks.entries()) {
                console.log(`  -> Generating embedding for chunk ${index + 1}/${chunks.length}...`);

                const vector = await createEmbedding(chunk);

                if (vector) {
                    points.push({
                        id: pointId++,
                        vector: vector,
                        payload: {
                            text: chunk,
                            source: file,
                            chunk_index: index,
                        },
                    });
                }
            }
        }

        // 3. Qdrant mein data daalna
        if (points.length > 0) {
            console.log(`\nInserting ${points.length} points into Qdrant...`);
            await qdrant.upsert(COLLECTION_NAME, { wait: true, batch: {
                ids: points.map(p => p.id),
                vectors: points.map(p => p.vector),
                payloads: points.map(p => p.payload),
            }});
            console.log(`\nIngestion complete! Total points: ${points.length}`);
        } else {
            console.log("No data points were created.");
        }

    } catch (error) {
        console.error("\nAn error occurred during ingestion:", error.message);
        console.log("\nMake sure Qdrant (on port 6336) is running and your 'chapters/' folder contains .md files.");
    }
}

ingestData();