import hashlib
import asyncio
import os
import joblib
from dotenv import load_dotenv
import google.genai as genai
from pinecone import PineconeAsyncio, ServerlessSpec
from tqdm import tqdm

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

class VectorDB:
    def __init__(self, index_name: str, batch_size: int = 100):
        self.embed_model = client.aio.models
        self.index_name = index_name
        self.pc = PineconeAsyncio(api_key=api_key)
        self.index = None
        self._initialised = False
        self.batch_size = batch_size

    async def ainit(self):
        if self._initialised:
            return self

        sample_embed = await self.embed_model.embed_content(
            model="embedding-001", 
            contents=["engineerLambda"]
        )
        embed_value = sample_embed.embeddings[0].values
        dim = len(embed_value)

        if not await self.pc.has_index(self.index_name):
            await self.pc.create_index(
                name=self.index_name,
                dimension=dim,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        desc = await self.pc.describe_index(self.index_name)
        index_host = desc.host
        self.index = self.pc.IndexAsyncio(host=index_host)

        self._initialised = True
        return self

    def generate_id(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def add_docs_to_store(self, data):
        if not self._initialised:
            await self.ainit()

        # Process in batches
        for start in tqdm(range(0, len(data), self.batch_size)):
            batch_df = data.iloc[start:start + self.batch_size]

            # Generate IDs for the batch
            ids = [self.generate_id(row['Description']) for _, row in batch_df.iterrows()]

            # Fetch existing IDs in bulk
            existing = await self.index.fetch(ids=ids)
            existing_ids = set(existing.vectors.keys()) if existing and existing.vectors else set()

            # Filter only new rows
            new_rows = [(i, row) for i, (idx, row) in enumerate(batch_df.iterrows()) if ids[i] not in existing_ids]
            if not new_rows:
                continue

            # Extract contents for embedding
            contents = [row['Description'] for _, row in new_rows]

            # Batch embed
            embeddings = await self.embed_model.embed_content(
                model="embedding-001",
                contents=contents
            )

            # Prepare vectors
            vectors = []
            for (i, row), emb in zip(new_rows, embeddings.embeddings):
                id_ = ids[i]
                vectors.append({
                    "id": id_,
                    "values": emb.values,
                    "metadata": {"description": row['Description'], "word": row["Word"]},
                })

            # Upsert batch
            if vectors:
                await self.index.upsert(vectors=vectors)

    async def query_store(self, query: str, k: int = 5):
        if not self._initialised:
            await self.ainit()

        query_embedding = await self.embed_model.embed_content(
            model="embedding-001",
            contents=[query]
        )

        results = await self.index.query(
            vector=query_embedding.embeddings[0].values,
            top_k=k,
            include_metadata=True
        )
        return results

async def main():
    data = joblib.load('resources/data.pkl')
    vectordb = VectorDB("reverse-dictionary", batch_size=100)
    await vectordb.ainit()
    await vectordb.add_docs_to_store(data)
    results = await vectordb.query_store(query="A person that works with computers", k=5)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
