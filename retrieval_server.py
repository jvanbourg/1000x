from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)

embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
vectorstore = Chroma(persist_directory="vectorstore_db", embedding_function=embeddings)

@app.route("/retrieve", methods=["POST"])
def retrieve():
    data = request.get_json()
    query = data.get("query", "")
    top_k = data.get("top_k", 3)
    results = vectorstore.similarity_search(query, k=top_k)
    chunks = [r.page_content for r in results]
    return jsonify({"chunks": chunks})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
