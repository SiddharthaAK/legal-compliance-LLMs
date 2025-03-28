from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from src.search import search_faiss, get_llm_response

# Initialize FastAPI app with rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()

# Add middleware for rate limiting
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

class QueryRequest(BaseModel):
    query: str

@app.post("/search/")
@limiter.limit("10/minute")  # ✅ Restrict to 10 requests per minute per IP
async def search_compliance(request: QueryRequest, request_ip: Request = Depends(get_remote_address)):
    retrieved_texts = search_faiss(request.query)
    summarized_text = "\n".join(retrieved_texts)
    refined_response = get_llm_response(request.query, summarized_text)

    return {"query": request.query, "response": refined_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
