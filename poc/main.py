from flask import Flask, request, jsonify
from service import create_agent, process_conversation, retrieve_similar_conversations
from prompts import ASSISTANT_PROMPT

app = Flask(__name__)

# Global dictionary to store per-user agents
user_chains = {}

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint for per-user conversation.
    Expects JSON with:
      - user_id: str (unique user identifier)
      - message: str (the new user message)
    Returns:
      - response: str (the agent's reply)
      
    If an agent exists for the user, it returns the LLM's response. Otherwise, it creates a new agent loaded with knowledge from Pinecone.
    """
    data = request.get_json()
    user_id = data.get("user_id")
    user_message = data.get("message")
    
    if not user_id or not user_message:
        return jsonify({"error": "user_id and message are required"}), 400

    # Check if an agent for this user exists
    if user_id not in user_chains:
        # Create a new agent by loading knowledge from Pinecone (if available)
        agent = create_agent(user_id)
        user_chains[user_id] = agent
    else:
        agent = user_chains[user_id]

    # Generate response using the user's conversation chain; the chain updates its memory automatically.
    response_text = agent.predict(input=user_message)
    
    return jsonify({"response": response_text})

@app.route('/process_conversation', methods=['POST'])
def process():
    """
    Endpoint to process a conversation:
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
    Endpoint to retrieve similar conversation summaries:
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
    # Serialize Document objects to JSON-serializable dicts
    serialized = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in similar] if similar else []
    return jsonify({"similar_conversations": serialized})

if __name__ == '__main__':
    app.run(debug=True)
