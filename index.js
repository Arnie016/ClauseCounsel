// 1. Import OpenAI library with Configuration and OpenAIApi
const { Configuration, OpenAIApi } = require("openai");
// 2. Import Pinecone database client
const { Pinecone } = require("@pinecone-database/pinecone");
// 3. Import dotenv for environment variable management
const dotenv = require("dotenv");
// 4. Load environment variables from .env file
dotenv.config();
// 5. Configuration for Pinecone and OpenAI
const config = {
  similarityQuery: {
    topK: 1, // Top results limit
    includeValues: false, // Exclude vector values
    includeMetadata: true, // Include metadata
  },
  namespace: "your-namespace", // Pinecone namespace
  indexName: "your-index-name", // Pinecone index name
  embeddingID: "your-embedding-id", // Embedding identifier
  dimension: 1536, // Embedding dimension
  metric: "cosine", // Similarity metric
  cloud: "aws", // Cloud provider
  region: "us-west-2", // Serverless region
  query: "What is my dog's name?", // Query example
};
// 6. Data to embed with modified metadata field
const dataToEmbed = [
  {
    textToEmbed: "My dog's name is Steve.",
    favouriteActivities: ["playing fetch", "running in the park"],
    born: "July 19, 2023",
  },
  {
    textToEmbed: "My cat's name is Sandy.",
    favouriteActivities: ["napping", "chasing laser pointers"],
    born: "August 7, 2019",
  },
];
// 7. Initialize OpenAI client with Configuration
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);
// 8. Initialize Pinecone client with API key
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
// 9. Function to store embeddings in Pinecone
async function storeEmbeddings() {
  // 10. Loop through each data item to embed
  await Promise.all(
    dataToEmbed.map(async (item, index) => {
      // 11. Create embedding using OpenAI
      const embeddingResponse = await openai.createEmbedding({
        model: "text-embedding-ada-002",
        input: item.textToEmbed,
      });
      const embedding = embeddingResponse.data.data[0].embedding;
      // 12. Define index name and unique ID for each embedding
      const indexName = config.indexName;
      const id = `${config.embeddingID}-${index + 1}`;
      // 13. Upsert embedding into Pinecone with new metadata
      await pc
        .index(indexName)
        .namespace(config.namespace)
        .upsert([
          {
            id: id,
            values: embedding,
            metadata: { ...item },
          },
        ]);
      // 14. Log embedding storage
      console.log(`Embedding ${id} stored in Pinecone.`);
    })
  );
}
// 15. Function to query embeddings in Pinecone
async function queryEmbeddings(queryText) {
  // 16. Create query embedding using OpenAI
  const queryEmbeddingResponse = await openai.createEmbedding({
    model: "text-embedding-ada-002",
    input: queryText,
  });
  const queryEmbedding = queryEmbeddingResponse.data.data[0].embedding;
  // 17. Perform the query
  const queryResult = await pc
    .index(config.indexName)
    .namespace(config.namespace)
    .query({
      ...config.similarityQuery,
      vector: queryEmbedding,
    });
  // 18. Log query results
  console.log(`Query: "${queryText}"`);
  console.log(`Result:`, queryResult);
  console.table(queryResult.matches);
}
// 19. Function to manage Pinecone index
async function manageIndex(action) {
  // 20. Check if index exists
  const indexExists = (await pc.listIndexes()).indexes.some((index) => index.name === config.indexName);
  // 21. Create or delete index based on action
  if (action === "create") {
    if (indexExists) {
      console.log(`Index '${config.indexName}' already exists.`);
    } else {
      await pc.createIndex({
        name: config.indexName,
        dimension: config.dimension,
        metric: config.metric,
        spec: { serverless: { cloud: config.cloud, region: config.region } },
      });
      console.log(`Index '${config.indexName}' created.`);
    }
  } else if (action === "delete") {
    if (indexExists) {
      await pc.deleteIndex(config.indexName);
      console.log(`Index '${config.indexName}' deleted.`);
    } else {
      console.log(`Index '${config.indexName}' does not exist.`);
    }
  } else {
    console.log('Invalid action specified. Use "create" or "delete".');
  }
}
// 22. Main function to orchestrate operations
async function main() {
  try {
    await manageIndex("create");
    await storeEmbeddings();
    await queryEmbeddings(config.query);
    // await manageIndex("delete");
  } catch (error) {
    console.error("An error occurred:", error);
  }
}
// 23. Run our main function
main().catch((error) => {
  console.error("Unhandled error in main function:", error);
});

const express = require('express');
const { spawn } = require('child_process');
const cors = require('cors');

const app = express();
const port = 3001;

app.use(express.json());
app.use(cors());

app.post('/ask', (req, res) => {
  const { question } = req.body;

  const pythonProcess = spawn('python', ['test.py']);

  pythonProcess.stdin.write(question + '\n');
  pythonProcess.stdin.end();

  let result = '';

  pythonProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Error: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      res.status(500).json({ error: 'An error occurred while processing the question.' });
    } else {
      res.json({ answer: result.trim() });
    }
  });
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

// Add this function to load the Langflow script
function loadLangflowScript() {
  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/gh/logspace-ai/langflow-embedded-chat@v1.0.6/dist/build/static/js/bundle.min.js';
  script.async = true;
  document.body.appendChild(script);
}

// Load the script when the component mounts
loadLangflowScript();

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
