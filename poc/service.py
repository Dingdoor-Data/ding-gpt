from utils import get_summary, upsert_embeddings, get_similar_embeddings, llm
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def create_agent(user_id):
    """
    Create a new conversation agent for the user.
    This function attempts to load past knowledge from Pinecone and pre-populate memory.
    Args:
        user_id (str): The unique identifier for the user.
    Returns:
        ConversationChain: A new conversation agent with memory.
    """
    # Use a generic query to load past conversation summaries for the user.
    query = "user conversation history"
    similar = get_similar_embeddings(user_id, query)
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    if similar:
        for doc in similar:
            # Pre-populate memory with retrieved summary; doc.page_content is assumed to be the summary text.
            memory.save_context({"input": "Load past context"}, {"output": doc.page_content})
    # Create a new ConversationChain for the user with the loaded memory.
    agent = ConversationChain(llm=llm, memory=memory)
    return agent

def process_conversation(user_id, conversation_history):
    """
    Process a conversation by generating a summary and upserting its embedding into Pinecone.
    Args:
        user_id (str): The user ID.
        conversation_history (list): The conversation history (list of message dicts).
    Returns:
        str: The generated summary.
    """
    summary = get_summary(conversation_history)
    upsert_embeddings(user_id, summary)
    return summary

def retrieve_similar_conversations(user_id, query):
    """
    Retrieve similar conversation summaries for the given query.
    Args:
        user_id (str): The user ID.
        query (str): The query text.
    Returns:
        list: List of similar conversation summaries.
    """
    return get_similar_embeddings(user_id, query)
