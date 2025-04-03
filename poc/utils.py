import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from pydantic import BaseModel, Field
from openai import OpenAI
from langmem import create_memory_manager

# Updated imports using langchain-community and langchain-openai packages
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone (using the new pinecone package)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = 'memory-db'
existing_indexes = [x['name'] for x in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(name=index_name, dimension=1536, metric="cosine", 
                    spec=ServerlessSpec(cloud='aws', region='us-west-2'))
index = pc.Index(index_name)

# Initialize OpenAI embeddings with langchain-openai
embedding_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the LangChain Pinecone vector store using the index instance
vector_store = LangchainPinecone(index=index, embedding=embedding_model, text_key='text')

# Initialize the chat model using langchain_community
llm = ChatOpenAI(model_name='gpt-4o-mini', openai_api_key=os.getenv("OPENAI_API_KEY"))

# IMPORTANT: Set memory_key to "history" to match prompt expectations
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Create the conversation chain
conversation = ConversationChain(llm=llm, memory=memory)

# Define a Summary schema to capture key insights
class Summary(BaseModel):
    """Record key insights about the customer from the interaction."""
    insights: str = Field(
        ...,
        description="What did you learn about the customer? What did you learn about yourself? 'I ...'"
    )

# Initialize the memory manager for generating summaries
manager = create_memory_manager(
    'gpt-4o-mini',
    schemas=[Summary],
    instructions="""
        You are a memory manager. Your job is to help the agent remember important information about the customer and the interaction.
        Be concise and summarize the key points of the user's input.
    """,
    enable_inserts=True
)

def get_summary(conversation_history):
    """
    Get a summary of the conversation using the memory manager.
    Args:
        conversation_history (list): List of messages (each a dict with 'role' and 'content').
    Returns:
        str: The summary insights extracted from the conversation.
    """
    summary_list = manager.invoke({"messages": conversation_history})
    # Extract the insights from the first summary object
    return summary_list[0].content.insights

def get_embeddings(text):
    """
    Get the embeddings for the text using the OpenAI API.
    Args:
        text (str): The text to embed.
    Returns:
        list: The embedding vector.
    """
    return embedding_model.embed_query(text)

def upsert_embeddings(user_id, summary):
    """
    Upsert the embeddings for the summary into the Pinecone index.
    Args:
        user_id (str): The user ID.
        summary (str): The summary text.
    """
    try:
        vector_store.add_texts(texts=[summary], metadatas=[{"user_id": user_id}])
        return True
    except Exception as e:
        print(f"Error upserting embeddings: {e}")
    return False

def get_similar_embeddings(user_id, query):
    """
    Retrieve similar summaries from the Pinecone index.
    Args:
        user_id (str): The user ID.
        query (str): The query text.
    Returns:
        list: List of similar summary documents.
    """
    try:
        results = vector_store.similarity_search(query, k=5, filter={"user_id": user_id})
        return results
    except Exception as e:
        print(f"Error getting similar embeddings: {e}")
    return None
