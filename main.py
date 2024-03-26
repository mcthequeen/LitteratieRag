#API modules
from fastapi import FastAPI


#Modules for VectorStore and inference with llm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from Qa_inference import Qa_inference



#Environ
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY") 



    
#FastAPI instance    
app = FastAPI()

# Initialize Supabase client
supabase_url = os.getenv("supabase_url")
supabase_anon_key = os.getenv("supabase_anon_key")
supabase_service_role_key = os.getenv("supabase_service_role_key")





#loading embedding for retriever
print("#Loading embedding retriever")
embedding = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-large")

#Loading db
db = FAISS.load_local("faiss", embedding)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.5})

#Instance for Qa_inference

qa = Qa_inference(retriever=retriever, api_key=api_key)



@app.get("/")
async def hello():
    return {"hello": "wooooooorld"}


@app.get('/Inference')
async def inference():
    user_query = "C'est quoi la maladie de crohn ?"
    qa.user_query += user_query
    documents_string, _ = qa.get_documents()

    response = await qa.get_llm_response(query=user_query, documents=documents_string)

    return response

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
