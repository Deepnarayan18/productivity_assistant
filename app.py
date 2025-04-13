from flask import Flask, render_template, request, jsonify
from backend.model import initialize_qa_chain
from backend.ingest_data import process_and_store_pdf
import os

app = Flask(__name__)

# Initialize QA chain when the app starts
qa_chain = initialize_qa_chain()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query', '')
    response = qa_chain({"query": query})
    return jsonify({
        "answer": response["result"],
        "sources": [doc.metadata for doc in response["source_documents"]]
    })

@app.route('/reload', methods=['POST'])
def reload_document():
    try:
        global qa_chain
        process_and_store_pdf()
        qa_chain = initialize_qa_chain()
        return jsonify({"status": "success", "message": "Document reloaded successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)