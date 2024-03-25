from fastapi import FastAPI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from Qa_inference import Qa_inference


app = FastAPI()


#loading embedding for retriever
print("#Loading embedding retriever")
embedding = HuggingFaceEmbeddings(model_name="dangvantuan/sentence-camembert-large")

#Loading db
db = FAISS.load_local("faiss", embedding )
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 15, "lambda_mult": 0.5})

#Instance for Qa_inference
api_key = "LtI5T8ieAIpJBcuUbcby4s7km38V3edo"
qa = Qa_inference(retriever=retriever, api_key=api_key)



@app.get("/")
async def hello():
    return {"hello": "wooooooorld"}

@app.get("/Inference")
async def Inference():
    user_query = "C'est quoi la maladie de crohn ?"
    qa.user_query += user_query
    documents_string, documents = qa.get_documents()

    response = await qa.get_llm_response(query=user_query, documents=documents_string)
    print(type(response))
    return response

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
