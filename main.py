#API modules
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from ratelimiter import RateLimiter

#Modules for VectorStore and inference with llm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from Qa_inference import Qa_inference


#Supabase module
import supabase




#Get API key for mistral
with open("API_KEY.txt", "r") as fichier:
    api_key = fichier.read()
    
    
#FastAPI instance    
app = FastAPI()

# Initialize Supabase client
supabase_url = ''
supabase_anon_key = ''
supabase_service_role_key = ''


# Define a rate limiter with a specific limit per user
rate_limiter = RateLimiter(max_calls=10, period='minute')

security = HTTPBearer()



#loading embedding for retriever
print("#Loading embedding retriever")
embedding = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-large")

#Loading db
db = FAISS.load_local("faiss", embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.5})

#Instance for Qa_inference

qa = Qa_inference(retriever=retriever, api_key=api_key)



@app.get("/")

async def hello():
    return {"hello": "wooooooorld"}

async def authenticate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        user = await supabase_client.auth.get_user(token)

        if user['error']:
            raise HTTPException(status_code=401, detail='Invalid token')

        user_id = user['user']['id']
        return user_id
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.post('/inference')
@rate_limiter.rate_limited('user:<user_id>')
async def inference(user_id: str = Depends(authenticate_token)):
    user_query = "C'est quoi la maladie de crohn ?"
    qa.user_query += user_query
    documents_string, _ = qa.get_documents()

    response = await qa.get_llm_response(query=user_query, documents=documents_string)

    return response

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
