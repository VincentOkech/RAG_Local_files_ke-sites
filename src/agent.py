from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from .vector_store import VectorStore
from .web_search import KenyanWebSearch
from .config import Config
import asyncio

class RAGAgent:
    def __init__(self):
        self.vector_store = VectorStore()
        self.web_search = KenyanWebSearch()
        
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create custom prompt template
        self.qa_prompt = PromptTemplate(
            template="""You are a helpful AI assistant focused on providing information about Kenya. 
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, say that you don't know and suggest searching for more information.
            Try to be specific and cite sources when possible.

            Context: {context}

            Chat History: {chat_history}
            
            Question: {question}

            Answer:""",
            input_variables=["context", "chat_history", "question"]
        )
        
        # Initialize the chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.vector_store.as_retriever(
                search_kwargs={"k": 4}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True,
            verbose=True
        )

    async def get_response(self, query: str) -> str:
        """
        Get a response for the given query.
        
        Args:
            query (str): The user's question
            
        Returns:
            str: The agent's response
        """
        try:
            # Try vector store first
            docs = self.vector_store.similarity_search(query)
            
            # If no relevant docs found, try web search
            if not docs:
                web_results = self.web_search.search(query)
                if web_results:
                    # Add web results to context
                    context = "\n".join([f"{r['title']}: {r['snippet']}" for r in web_results])
                    query = f"Context from web search: {context}\nQuestion: {query}"
            
            # Run the chain
            response = await asyncio.to_thread(
                self.chain,
                {"question": query}
            )
            
            # Extract answer and sources
            answer = response.get("answer", "I couldn't find an answer to your question.")
            sources = response.get("source_documents", [])
            
            # Format response with sources if available
            if sources:
                source_texts = [f"Source {i+1}: {doc.metadata.get('source', 'Unknown source')}"
                              for i, doc in enumerate(sources)]
                answer = f"{answer}\n\nSources:\n" + "\n".join(source_texts)
            
            return answer
            
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def add_documents(self, documents):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        try:
            self.vector_store.add_documents(documents)
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False 