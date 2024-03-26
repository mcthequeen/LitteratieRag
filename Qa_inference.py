from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from sse_starlette.sse import EventSourceResponse


class Qa_inference():
    
    def __init__(self, retriever, api_key):
        self.retriever = retriever
        self.api_key = api_key
        self.history =""
        self.user_query = ""
        
    def get_documents(self):
        query = self.user_query
        """
        Users's query as input.
        invokes retriever and store the results in documents (format document from langchain). 
        documents_string is the name and page content for each doc.
        """
        documents = self.retriever.invoke(query)
        
        documents_string = ''
        for doc in documents:
            documents_string +="Document :" + doc.metadata["name"] +'\n'
            documents_string += doc.page_content + '\n'
            
        return documents_string, documents
    
    
    async def get_llm_response(self, query, documents):
        model = "open-mixtral-8x7b"

        client = MistralAsyncClient(api_key=self.api_key)
        pre_prompt = f"""Vous êtes des assistants santé. Répondez aux questions des utilisateurs en utilisant seulement les documents fournis. Si vous ne trouvez pas la réponse, dites que vous ne savez pas.
        Voici l'historique des conversations :
        {self.history}

        Voici les documents :
        {documents}

        Vulgarisez les termes médicaux le plus possible.
        Il est inutile d'indiquer aux utilisateurs de discuter ou demander conseil àleur médecin ou leur équipe médicale.
        Soyez polis et rassurants.
        Utilisez les informations pour répondre aux utilisateurs.
        """
        

        messages = [
            ChatMessage(role="system", content=pre_prompt.format(documents=documents)),
            ChatMessage(role="user", content=query)
        ]

        # With streaming
        async_response = client.chat_stream(model=model, messages=messages)

        async def event_generator(self):
            all_content =""
            async for chunk in async_response:
                content = chunk.choices[0].delta.content
                if content:
                    all_content += content
                    yield f'data: {{"text": "{all_content}"}}\n\n'
                   
            self.history += "User :" + query + "\n"
            self.history += "Assistant :" + all_content + "\n"
        return EventSourceResponse(event_generator(self))