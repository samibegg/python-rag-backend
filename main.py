import os
import tempfile
import uuid
import logging
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware # For allowing requests from your Next.js frontend
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
import openai # Keep this for general openai types if needed, but client will be AsyncOpenAI
from openai import AsyncOpenAI # Import AsyncOpenAI
from huggingface_hub import InferenceClient
import python_multipart # Ensure this is installed, though often not directly imported

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Variable Loading ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not found.")
if not HF_TOKEN:
    logger.warning("HUGGING_FACE_HUB_TOKEN environment variable not found.")

# --- FastAPI App Initialization ---
app = FastAPI(title="RAG AI Backend")

# --- CORS Middleware ---
origins = [
    "http://localhost:3000", # Your Next.js local dev URL
    "https://forgemission.com", # Your deployed frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- Global Variables & Configuration ---
embedding_models_cache = {}
FAISS_INDEX_DIR = tempfile.gettempdir()
logger.info(f"FAISS index and chunks will be stored in: {FAISS_INDEX_DIR}")

# --- Pydantic Models for Request/Response ---
class SetupResponse(BaseModel):
    message: str
    session_id: str
    num_chunks: int
    embedding_model_used: str

class QueryRequest(BaseModel):
    prompt: str
    llm_model: str
    embedding_model: str | None = None 
    session_id: str | None = None      
    use_rag: bool = False

class QueryResponse(BaseModel):
    result: str
    source: str 

# --- LLM Client Initialization ---
# Use AsyncOpenAI for awaitable calls
openai_client: AsyncOpenAI | None = None # Type hint for clarity
if OPENAI_API_KEY:
    try:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) # Initialize AsyncOpenAI
        logger.info("AsyncOpenAI client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize AsyncOpenAI client: {e}")

hf_inference_client = None
if HF_TOKEN:
    try:
        hf_inference_client = InferenceClient(token=HF_TOKEN)
        logger.info("Hugging Face InferenceClient initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Hugging Face InferenceClient: {e}")

# --- Helper Functions ---
def get_embedding_model_instance(model_name: str):
    if model_name not in embedding_models_cache:
        logger.info(f"Loading embedding model: {model_name}")
        try:
            embedding_models_cache[model_name] = SentenceTransformer(model_name)
            logger.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading embedding model {model_name}: {str(e)}")
    return embedding_models_cache[model_name]

def chunk_text_content(text: str, chunk_size: int = 300, chunk_overlap: int = 30):
    words = text.split()
    chunks = []
    current_pos = 0
    while current_pos < len(words):
        end_pos = current_pos + chunk_size
        chunk_words = words[current_pos:end_pos]
        chunks.append(" ".join(chunk_words))
        if end_pos >= len(words):
            break
        current_pos += (chunk_size - chunk_overlap)
    return [chunk for chunk in chunks if chunk.strip()]

# --- API Endpoints ---
@app.post("/setup-rag/", response_model=SetupResponse)
async def setup_rag_endpoint(
    file: UploadFile = File(...),
    embedding_model: str = Form(...)
):
    logger.info(f"Received /setup-rag/ request. Embedding model: {embedding_model}, File: {file.filename}")
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .txt files are supported.")

    try:
        contents = await file.read()
        try:
            text_content = contents.decode('utf-8')
        except UnicodeDecodeError:
            text_content = contents.decode('latin-1') 

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        text_chunks = chunk_text_content(text_content)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="No text chunks generated.")
        
        logger.info(f"Generated {len(text_chunks)} text chunks.")

        sbert_model = get_embedding_model_instance(embedding_model)
        embedding_dim = sbert_model.get_sentence_embedding_dimension()

        logger.info(f"Creating embeddings for {len(text_chunks)} chunks (dim: {embedding_dim})...")
        chunk_embeddings = sbert_model.encode(text_chunks, show_progress_bar=False)
        chunk_embeddings_np = np.array(chunk_embeddings).astype('float32')

        index = faiss.IndexFlatL2(embedding_dim)
        index.add(chunk_embeddings_np)
        logger.info(f"FAISS index created. Total vectors: {index.ntotal}")

        session_id = str(uuid.uuid4())
        index_filename = f"faiss_index_{session_id}.idx"
        index_filepath = os.path.join(FAISS_INDEX_DIR, index_filename)
        chunks_filename = f"text_chunks_{session_id}.txt"
        chunks_filepath = os.path.join(FAISS_INDEX_DIR, chunks_filename)

        faiss.write_index(index, index_filepath)
        with open(chunks_filepath, "w", encoding="utf-8") as f:
            for chunk in text_chunks:
                f.write(chunk.replace("\n", " ") + "\n")
        
        logger.info(f"FAISS index saved to {index_filepath}, chunks to {chunks_filepath} for session {session_id}")
        
        return SetupResponse(
            message="Dataset processed and FAISS index created successfully.",
            session_id=session_id,
            num_chunks=len(text_chunks),
            embedding_model_used=embedding_model
        )
    except Exception as e:
        logger.error(f"Error in /setup-rag/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/query-rag/", response_model=QueryResponse)
async def query_rag_endpoint(query_data: QueryRequest):
    logger.info(f"Received /query-rag/ request: {query_data.model_dump_json(indent=2)}")

    if not query_data.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    if not query_data.llm_model:
        raise HTTPException(status_code=400, detail="LLM model ID cannot be empty.")

    final_response_content = ""
    response_source = "direct_llm"
    current_prompt_to_llm = query_data.prompt # Default to original prompt

    if query_data.use_rag:
        logger.info("Processing RAG query.")
        if not query_data.session_id or not query_data.embedding_model:
            raise HTTPException(status_code=400, detail="session_id and embedding_model are required for RAG.")

        index_filepath = os.path.join(FAISS_INDEX_DIR, f"faiss_index_{query_data.session_id}.idx")
        chunks_filepath = os.path.join(FAISS_INDEX_DIR, f"text_chunks_{query_data.session_id}.txt")

        if not os.path.exists(index_filepath) or not os.path.exists(chunks_filepath):
            logger.error(f"FAISS index or chunks not found for session_id: {query_data.session_id}")
            raise HTTPException(status_code=404, detail="Vectorized data not found. Please upload and vectorize first.")

        try:
            logger.info(f"Loading FAISS index from {index_filepath}")
            index = faiss.read_index(index_filepath)
            
            logger.info(f"Loading text chunks from {chunks_filepath}")
            with open(chunks_filepath, "r", encoding="utf-8") as f:
                all_text_chunks = [line.strip() for line in f.readlines()]

            sbert_model = get_embedding_model_instance(query_data.embedding_model)
            logger.info(f"Embedding query prompt with {query_data.embedding_model}...")
            query_embedding = sbert_model.encode([query_data.prompt])
            query_embedding_np = np.array(query_embedding).astype('float32')

            k = 3 
            logger.info(f"Searching FAISS index for top {k} chunks...")
            distances, indices = index.search(query_embedding_np, k)
            
            retrieved_context = ""
            if len(indices[0]) > 0 and indices[0][0] != -1 : 
                logger.info(f"Retrieved indices: {indices[0]}")
                for i_val in indices[0]: # Renamed i to i_val to avoid conflict if openai was imported as i
                    if 0 <= i_val < len(all_text_chunks): 
                        retrieved_context += all_text_chunks[i_val] + "\n\n"
            
            if not retrieved_context.strip():
                logger.info("No relevant context found from RAG. Using original prompt for LLM.")
                # current_prompt_to_llm is already query_data.prompt
            else:
                logger.info(f"Retrieved context (first 100 chars): {retrieved_context[:100]}...")
                current_prompt_to_llm = f"Based on the following context:\n\n{retrieved_context}\n\nAnswer the question: {query_data.prompt}"
                response_source = "rag"
            
        except Exception as e:
            logger.error(f"Error during RAG processing for session {query_data.session_id}: {e}", exc_info=True)
            # Fallback to direct LLM if RAG fails; current_prompt_to_llm is already original prompt
            response_source = "direct_llm_after_rag_error"
    else:
        logger.info("Processing direct LLM query (non-RAG).")
        # current_prompt_to_llm is already query_data.prompt
        response_source = "direct_llm"

    # Call the LLM
    try:
        logger.info(f"Calling LLM: {query_data.llm_model} with prompt (first 100 chars): {current_prompt_to_llm[:100]}...")
        if query_data.llm_model.startswith("gpt-"):
            if not openai_client: # Check if client is initialized
                logger.error("AsyncOpenAI client not initialized. Cannot make OpenAI call.")
                raise HTTPException(status_code=503, detail="OpenAI service not configured or API key missing.")
            
            # This call is now correctly using await with an async client method
            chat_completion = await openai_client.chat.completions.create(
                messages=[{"role": "user", "content": current_prompt_to_llm}],
                model=query_data.llm_model,
            )
            final_response_content = chat_completion.choices[0].message.content
        
        elif query_data.llm_model.startswith("huggingfaceh4/") or query_data.llm_model.startswith("mistralai/"):
            if not hf_inference_client:
                logger.error("Hugging Face client not initialized. Cannot make HF call.")
                raise HTTPException(status_code=503, detail="Hugging Face service not configured or API token missing.")
            
            # Note: hf_inference_client.text_generation is synchronous.
            # If you need async for HF, you'd typically wrap it or use an async HTTP client.
            # For simplicity, keeping it synchronous here. FastAPI handles running sync code in a threadpool.
            final_response_content = hf_inference_client.text_generation(
                prompt=current_prompt_to_llm, model=query_data.llm_model, max_new_tokens=500
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported LLM model ID: {query_data.llm_model}")
        
        logger.info(f"LLM call successful. Response source: {response_source}")
        return QueryResponse(result=final_response_content, source=response_source)

    except openai.APIError as e: # Catch specific OpenAI errors
        logger.error(f"OpenAI API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}") # Use str(e) for detail
    except Exception as e: # Catch other errors
        logger.error(f"Error calling LLM {query_data.llm_model}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error with LLM {query_data.llm_model}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

