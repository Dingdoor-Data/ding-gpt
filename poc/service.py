from utils import get_summary, upsert_embeddings, get_similar_embeddings

def process_conversation(user_id, conversation_history):
    """
    Process a conversation by generating a summary and upserting its embeddings into Pinecone.
    Args:
        user_id (str): The user ID.
        conversation_history (list): List of messages (dicts with 'role' and 'content').
    Returns:
        str: The generated conversation summary.
    """
    summary = get_summary(conversation_history)
    upsert_embeddings(user_id, summary)
    return summary

def retrieve_similar_conversations(user_id, query):
    """
    Retrieve similar conversation summaries for a given query.
    Args:
        user_id (str): The user ID.
        query (str): The query text.
    Returns:
        list: A list of similar conversation summaries.
    """
    return get_similar_embeddings(user_id, query)
