#API modules
from fastapi import FastAPI
from supabase import create_client
from pydantic import BaseModel

#Modules for VectorStore and inference with llm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from Qa_inference import Qa_inference


#Environ
from decouple import config 
import os

#FastAPI instance    
app = FastAPI()

supabase_url = config("SUPABASE_URL")

print(supabase_url)
#supabase_anon_key = os.getenv("supabase_anon_key")
supabase_service_role_key = config("SUPABASE_SERVICE_ROLE_KEY")

#supabase = create_client(supabase_url, supabase_anon_key)
admin_supabase = create_client(supabase_url,supabase_service_role_key)


#loading embedding for retriever
print("#Loading embedding retriever")
embedding = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-large")

#Loading db
db = FAISS.load_local("faiss", embedding, allow_dangerous_deserialization =True)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.5})

#Instance for Qa_inference
api_key = os.getenv("MISTRAL_API_KEY") 
qa = Qa_inference(retriever=retriever, api_key=api_key)




def verify_jwt(jwt):
    user_data = admin_supabase.auth.get_user(jwt)
    if user_data:
        return True
    else:
        False

@app.get("/")
async def hello():
    return {"hello": "wooooooorld"}


class UserCreate(BaseModel):
    jwt: str
    query: str

#data : UserCreate
@app.get('/inference')
async def inference():
    #user_query= data.query
    user_query = "C'est quoi le covid ?"
    print(user_query)
    #jwt = data.jwt
    #verify_jwt(jwt)
    On = True
    if On:
        qa.user_query += user_query
        documents_string, _ = qa.get_documents()

        response = await qa.get_llm_response(query=user_query, documents=documents_string)
        print(response)

        return response
    else:
        return {"JWT incorect"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
