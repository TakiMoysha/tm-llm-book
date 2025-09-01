- Embedding model: A pre-trained language model that converts input text into embeddings - vector representations that capture semantic meaning. These vectors will be used to search for relevant information in the dataset.
- Vector database: A storage system for knowledge and its corresponding embedding vectors. While there are many vector database technologies like Qdrant, Pinecone, and pgvector, we'll implement a simple in-memory database from scratch.
- Chatbot: A language model that generates responses based on retrieved knowledge. This can be any language model, such as Llama, Gemma, or GPT.

**Indexing phase** - step in creating a rag system. Это разбиение данных на чанки и вычисления векторного представления для каждого чанка, который можно эффективно искать во время генерации.

> Мне интересны qdrant и chromadb.

### Stack

- langchain — для сборки RAG-пайплайна.
- sentence-transformers — для генерации эмбеддингов.
- chromadb — локальная векторная БД.
- requests — для взаимодействия с LM Studio через API.

Модель - Mistral-7B-instruct-v0.3

### 
