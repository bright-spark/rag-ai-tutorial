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
		"RecursiveCharacterTextSplitter generated ${texts.length} chunks",
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
		  },
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
		  },
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
	  },
	);
  
    // Return the final answer
	return c.text(answer);
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
