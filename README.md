Run locally:

git clone
cd 
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000


test:
--
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "What is AI?",
        "llm_model": "gpt-3.5-turbo",
        "use_rag": false
      }' \

--
curl -X POST \
-F "file=@path/to/file.txt" \
-F "embedding_model=sentence-transformers/all-MiniLM-L6-v2" \
http://localhost:8000/setup-rag/

# will return session-id


curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "What is AI?",
        "llm_model": "gpt-3.5-turbo",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "session_id": "session-id",
        "use_rag": true
      }' \


