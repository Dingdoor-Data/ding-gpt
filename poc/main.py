from flask import Flask, request, jsonify
from service import process_conversation, retrieve_similar_conversations
from prompts import ASSISTANT_PROMPT
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

app = Flask(__name__)

# Global dictionary to hold conversation chains per user
user_chains = {}

# Initialize the LLM (this can be shared across users)
llm = ChatOpenAI(model_name='gpt-4o-mini', openai_api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint for per-user conversation.
    Expects JSON with:
      - user_id: str
      - message: str (the new user message)
    Returns:
      - response: str (the agent's reply)
      
    Each user gets their own ConversationChain with separate memory.
    """
    data = request.get_json()
    user_id = data.get("user_id")
    user_message = data.get("message", "")

    if not user_id or not user_message:
        return jsonify({"error": "user_id and message are required"}), 400

    # Retrieve or create the user's conversation chain
    if user_id not in user_chains:
        # Create a new ConversationBufferMemory and ConversationChain for the user
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        user_chains[user_id] = ConversationChain(llm=llm, memory=memory)
    
    chain = user_chains[user_id]
    
    # Generate response using the user's conversation chain (memory is updated automatically)
    response_text = chain.predict(input=user_message)
    
    return jsonify({"response": response_text})

@app.route('/process_conversation', methods=['POST'])
def process():
    """
    Endpoint to process a conversation (for generating summaries, etc.).
    Expects JSON with:
      - user_id: str
      - conversation_history: list of messages (each with 'role' and 'content')
    Returns the generated summary.
    """
    data = request.get_json()
    user_id = data.get('user_id')
    conversation_history = data.get('conversation_history', [])
    if not user_id or not conversation_history:
        return jsonify({"error": "user_id and conversation_history are required"}), 400

    summary = process_conversation(user_id, conversation_history)
    return jsonify({"summary": summary})

@app.route('/retrieve_similar', methods=['POST'])
def retrieve():
    """
    Endpoint to retrieve similar conversation summaries.
    Expects JSON with:
      - user_id: str
      - query: str
    Returns a list of similar conversation summaries.
    """
    data = request.get_json()
    user_id = data.get('user_id')
    query = data.get('query')
    if not user_id or not query:
        return jsonify({"error": "user_id and query are required"}), 400

    similar = retrieve_similar_conversations(user_id, query)
    # Convert Document objects to serializable dictionaries
    serialized_docs = [
        {"page_content": doc.page_content, "metadata": doc.metadata} 
        for doc in similar
    ] if similar else []
    
    return jsonify({"similar_conversations": serialized_docs})


if __name__ == '__main__':
    app.run(debug=True)
