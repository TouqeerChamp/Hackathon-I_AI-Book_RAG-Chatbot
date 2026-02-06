require('dotenv').config();
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { QdrantClient } = require('@qdrant/js-client-rest');

async function testFullFlow() {
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const qdrantClient = new QdrantClient({
        url: process.env.QDRANT_URL,
        apiKey: process.env.QDRANT_API_KEY,
    });

    const COLLECTION_NAME = 'humanoid_robotics_textbook';
    const EMBEDDING_MODEL = 'text-embedding-004';

    console.log('🔍 Testing embedding generation...');
    const model = genAI.getGenerativeModel({ model: EMBEDDING_MODEL });
    const result = await model.embedContent("What is humanoid robotics?");
    const queryEmbedding = result.embedding.values;
    console.log('✅ Embedding dimensions:', queryEmbedding.length);

    console.log('🔍 Testing Qdrant search...');
    try {
        // Check collection info first
        const info = await qdrantClient.getCollection(COLLECTION_NAME);
        console.log('✅ Collection info:', {
            name: info.name,
            points_count: info.points_count,
            vector_size: info.config.params.vectors.size
        });

        // Test search
        const searchResults = await qdrantClient.search(COLLECTION_NAME, {
            vector: queryEmbedding,
            limit: 3,
            with_payload: true
        });

        console.log('✅ Search results:', searchResults.length, 'chunks found');
        searchResults.forEach((r, i) => {
            console.log(`  ${i+1}. Score: ${r.score}, Chunk: ${r.payload.text.substring(0, 50)}...`);
        });

    } catch (error) {
        console.error('❌ Qdrant error:', error.message);
        console.error('Stack:', error.stack);
    }
}

testFullFlow().catch(console.error);
