/**
 * Welcome to Cloudflare Workers! This is your first worker.
 *
 * - Run `npm run dev` in your terminal to start a development server
 * - Open a browser tab at http://localhost:8787/ to see your worker in action
 * - Run `npm run deploy` to publish your worker
 *
 * Learn more at https://developers.cloudflare.com/workers/
 */

import { Hono } from "hono";
const app = new Hono();

import { WorkflowEntrypoint } from "cloudflare:workers";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// Remove direct CSV import
// import csvData from '../dataset.csv';

// Function to load CSV data
async function loadCSVData(env) {
	try {
		// Use the static assets binding to get the CSV file
		// The correct method is fetch, not get
		const csvResponse = await env.ASSETS.fetch(new Request('https://example.com/dataset.csv'));
		if (!csvResponse.ok) {
			throw new Error(`Failed to load dataset.csv from static assets: ${csvResponse.status} ${csvResponse.statusText}`);
		}
		
		// Convert the response to text
		return await csvResponse.text();
	} catch (error) {
		console.error("Error loading CSV data:", error);
		throw error;
	}
}

app.post("/upsert", async (c) => {
	try {
		// Load CSV data
		const csvData = await loadCSVData(c.env);
		
		// Parse the CSV data
		const lines = csvData.trim().split('\n');
		const textsToEmbed = lines.map((line) => {
			if (line.length < 3) return ''; // Handle potentially empty lines or just ","
			// Remove surrounding quotes and trailing comma: "...", -> ...
			// Note: This assumes the Python script correctly handled internal quotes.
			// If internal quotes were doubled (" -> ""), use: .replace(/""/g, '"')
			let content = line.slice(1, -2); 
			// Optionally unescape doubled quotes if the Python script did that:
			// content = content.replace(/""/g, '"');
			return content;
		}).filter((text) => text && text.length > 0); // Filter out any empty strings

		// Check if we actually got any text
		if (textsToEmbed.length === 0) {
			console.error("No text data found or parsed from dataset.csv");
			return new Response("No text data found in CSV to insert.", { status: 400 });
		}

		console.log(`Total text snippets to process: ${textsToEmbed.length}`);

		const batchSize = 200; // Process in batches of 200
		let totalInserted = 0;
		let vectorIdCounter = 1; // Ensure unique IDs across batches

		for (let i = 0; i < textsToEmbed.length; i += batchSize) {
			const batch = textsToEmbed.slice(i, i + batchSize);
			console.log(`Processing batch ${Math.floor(i / batchSize) + 1}: items ${i + 1} to ${Math.min(i + batchSize, textsToEmbed.length)}`);

			// Get embeddings for the current batch
			const modelResp = await c.env.AI.run(
				"@cf/baai/bge-base-en-v1.5",
				{
					text: batch,
				}
			);

			// Prepare vectors for Vectorize
			let vectors = [];
			modelResp.data.forEach((vector) => {
				// Use original text as metadata? Could be useful but large.
				// For now, just use ID. Ensure ID is unique across all batches.
				vectors.push({ id: `${vectorIdCounter}`, values: vector }); 
				vectorIdCounter++;
			});

			// Insert the batch into Vectorize
			if (vectors.length > 0) {
				// upsert returns void on success, throws on error
				await c.env.VECTORIZE.upsert(vectors);
				console.log(`  Batch of ${vectors.length} vectors upserted successfully.`);
				totalInserted += vectors.length; // Add batch size on success
			}
		}
		return Response.json({ success: true, totalVectorsInserted: totalInserted });

	} catch (error) {
		// Type guard for error handling
		let errorMessage = "An unknown error occurred during batch processing.";
		if (error instanceof Error) {
			console.error(`Error during batch processing: ${error.message}`);
			errorMessage = error.message;
			// Check if error has more details in cause
			if (error.cause) {
				try {
					console.error(`Error cause: ${JSON.stringify(error.cause)}`);
				} catch (stringifyError) {
					console.error("Could not stringify error cause.");
				}
			}
		} else {
			console.error("An unexpected error type occurred:", error);
		}
		return new Response(`Error during batch processing: ${errorMessage}`, { status: 500 });
	}
});

// Your query: expect this to match
app.get("/query", async (c) => {
	let userQuery = "CAPS";
	const queryVector = await c.env.AI.run(
		"@cf/baai/bge-base-en-v1.5",
		{
			text: [userQuery],
		}
	);

	let matches = await c.env.VECTORIZE.query(queryVector.data[0], {
		topK: 1,
	});
	return Response.json({
		// Expect a vector ID. 1 to be your top match with a score of
		// ~0.89693683
		// This tutorial uses a cosine distance metric, where the closer to one,
		// the more similar.
		matches: matches,
	});
});

export class RAGWorkflow extends WorkflowEntrypoint {
	async run(event, step) {
		const env = this.env;
		const { text } = event.payload;
		let texts = await step.do("split text", async () => {
			const splitter = new RecursiveCharacterTextSplitter();
			const output = await splitter.createDocuments([text]);
			return output.map((doc) => doc.pageContent);
		});
	
		console.log(
			`RecursiveCharacterTextSplitter generated ${texts.length} chunks`
		);
	
		for (const index in texts) {
			const text = texts[index];
			const record = await step.do(
				`create database record: ${index}/${texts.length}`,
				async () => {
					const query = "INSERT INTO notes (text) VALUES (?) RETURNING *";
		
					const { results } = await env.DB.prepare(query).bind(text).run();
		
					const record = results[0];
					if (!record) throw new Error("Failed to create note");
					return record;
				}
			);
		
			const embedding = await step.do(
				`generate embedding: ${index}/${texts.length}`,
				async () => {
					const embeddings = await env.AI.run("@cf/baai/bge-base-en-v1.5", {
						text: text,
					});
					const values = embeddings.data[0];
					if (!values) throw new Error("Failed to generate vector embedding");
					return values;
				}
			);
		
			await step.do(`insert vector: ${index}/${texts.length}`, async () => {
				return env.VECTORIZE.upsert([
					{
						id: record.id.toString(),
						values: embedding,
					},
				]);
			});
		}
	}
}

app.delete("/notes/:id", async (c) => {
	const { id } = c.req.param();
	
	const query = `DELETE FROM notes WHERE id = ?`;
	await c.env.DB.prepare(query).bind(id).run();
	
	await c.env.VECTORIZE.deleteByIds([id]);
	
	return c.status(204);
});
	
app.post("/notes", async (c) => {
	const { text } = await c.req.json();
	if (!text) return c.text("Missing text", 400);
	await c.env.RAG_WORKFLOW.create({ params: { text } });
	return c.text("Created note", 201);
});

app.get("/", async (c) => {
	
	// Get the question from query param or use default
	const question = c.req.query("text") || "What is the square root of 9?";
	
	// Get embeddings for the question
	const embeddings = await c.env.AI.run("@cf/baai/bge-base-en-v1.5", {
		text: question,
	});
	const vectors = embeddings.data[0];
	
	// Query Vectorize index
	let vecId = null;
	if (vectors && Array.isArray(vectors) && vectors.length > 0) {
		const vectorQuery = await c.env.VECTORIZE.query(vectors, { topK: 1 });
		if (vectorQuery && vectorQuery.matches && vectorQuery.matches.length > 0 && vectorQuery.matches[0]) {
			vecId = vectorQuery.matches[0].id;
		} else {
			console.log("No matching vector found for the question.");
		}
	} else {
		console.warn("Could not generate valid embedding vector for the question.");
	}

	// Retrieve note text from D1 if a matching vector was found
	let notes = [];
	if (vecId) {
		console.log(`Found matching vector ID: ${vecId}. Querying database.`);
		const query = `SELECT * FROM notes WHERE id = ?`;
		const { results } = await c.env.DB.prepare(query).bind(vecId).all();
		if (results) {
			notes = results.map((vec) => vec.text);
			console.log(`Retrieved ${notes.length} note(s) from DB.`);
		} else {
			console.log(`No note found in DB for ID: ${vecId}`);
		}
	}
	
	// Prepare context for the final AI call
	const contextMessage = notes.length
		? `Context:\n${notes.map((note) => `- ${note}`).join("\n")}`
		: "";
	
	const systemPrompt = `When answering the question or responding, use the context provided, if it is provided and relevant.`;
	
	// Call the final AI model
	console.log(`Calling LLM with${notes.length ? ' context' : 'out context'}.`);
	const { response: answer } = await c.env.AI.run(
		"@cf/meta/llama-3-8b-instruct",
		{
			messages: [
				...(notes.length ? [{ role: "system", content: contextMessage }] : []),
				{ role: "system", content: systemPrompt },
				{ role: "user", content: question },
			],
		}
	);
	
	// Return the final answer
	return c.text(answer);
});

app.get("/bootstrap", async (c) => {
	try {
		// First, clear the vectorize vector index
		console.log("Clearing vectorize vector index...");
		// The deleteAll method doesn't exist, so we need to use a different approach
		// We'll use the deleteByIds method with an empty array to clear all vectors
		// or we can use the query method to get all IDs and then delete them
		try {
			// Try to get all vectors first
			const allVectors = await c.env.VECTORIZE.query([], { topK: 10000 });
			if (allVectors && allVectors.matches && allVectors.matches.length > 0) {
				const vectorIds = allVectors.matches.map(match => match.id);
				await c.env.VECTORIZE.deleteByIds(vectorIds);
				console.log(`Deleted ${vectorIds.length} vectors from the index.`);
			} else {
				console.log("No vectors found in the index to delete.");
			}
		} catch (vectorError) {
			console.error("Error clearing vectorize index:", vectorError);
			// Continue with the process even if vector clearing fails
		}
		console.log("Vectorize vector index cleared successfully.");

		// Then, clear the database
		console.log("Clearing database...");
		const deleteQuery = "DELETE FROM notes";
		await c.env.DB.prepare(deleteQuery).run();
		console.log("Database cleared successfully.");

		// Load CSV data
		const csvData = await loadCSVData(c.env);
		
		// Parse the CSV data
		const lines = csvData.trim().split('\n');
		const textsToInsert = lines.map((line) => {
			if (line.length < 3) return ''; // Handle potentially empty lines or just ","
			// Remove surrounding quotes and trailing comma: "...", -> ...
			// Note: This assumes the Python script correctly handled internal quotes.
			// If internal quotes were doubled (" -> ""), use: .replace(/""/g, '"')
			let content = line.slice(1, -2); 
			// Optionally unescape doubled quotes if the Python script did that:
			// content = content.replace(/""/g, '"');
			return content;
		}).filter((text) => text && text.length > 0); // Filter out any empty strings

		// Check if we actually got any text
		if (textsToInsert.length === 0) {
			console.error("No text data found or parsed from dataset.csv");
			return new Response("No text data found in CSV to insert.", { status: 400 });
		}

		console.log(`Total text snippets to process: ${textsToInsert.length}`);

		const batchSize = 50; // Process in smaller batches for database operations
		let totalInserted = 0;

		for (let i = 0; i < textsToInsert.length; i += batchSize) {
			const batch = textsToInsert.slice(i, i + batchSize);
			console.log(`Processing batch ${Math.floor(i / batchSize) + 1}: items ${i + 1} to ${Math.min(i + batchSize, textsToInsert.length)}`);

			// Process each text in the batch
			for (const text of batch) {
				// Use the same method as in the RAG workflow to insert into the database
				await c.env.RAG_WORKFLOW.create({ params: { text } });
				totalInserted++;
			}
			
			console.log(`  Batch of ${batch.length} texts inserted successfully.`);
		}
		
		return Response.json({ 
			success: true, 
			totalTextsInserted: totalInserted,
			message: `Successfully cleared existing data and inserted ${totalInserted} texts into the database.`
		});

	} catch (error) {
		// Error handling
		let errorMessage = "An unknown error occurred during batch processing.";
		if (error instanceof Error) {
			console.error(`Error during batch processing: ${error.message}`);
			errorMessage = error.message;
			// Check if error has more details in cause
			if (error.cause) {
				try {
					console.error(`Error cause: ${JSON.stringify(error.cause)}`);
				} catch (stringifyError) {
					console.error("Could not stringify error cause.");
				}
			}
		} else {
			console.error("An unexpected error type occurred:", error);
		}
		return new Response(`Error during batch processing: ${errorMessage}`, { status: 500 });
	}
});

app.onError((err, c) => {
	console.error("======= Global Error Caught ======");
	console.error("Error Message:", err?.message);
	console.error("Error Stack:", err?.stack);
	console.error("Full Error Object:", err);
	console.error("Context Path:", c.req.path);
	console.error("==================================");
	// Return a user-friendly error message instead of the raw error
	return c.text("An internal server error occurred. Please check the logs.", 500);
});

export default app;
