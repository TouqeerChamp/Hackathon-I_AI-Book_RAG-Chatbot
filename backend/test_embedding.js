require('dotenv').config();
const { GoogleGenerativeAI } = require('@google/generative-ai');

async function testEmbedding() {
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: 'text-embedding-004' });

    const result = await model.embedContent("Test text");
    const vector = result.embedding.values;
    console.log('Vector dimensions:', vector.length);
    console.log('First 5 values:', vector.slice(0, 5));
}

testEmbedding().catch(console.error);
