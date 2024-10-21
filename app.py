from flask import Flask, render_template, request, jsonify
from together import Together
import os
import json
from pydantic import BaseModel, Field
from typing import List
from graphviz import Digraph
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in .env file")

class Node(BaseModel):
    id: str
    label: str

class Edge(BaseModel):
    source: str
    target: str
    label: str

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

def generate_graph(input) -> KnowledgeGraph:
    together = Together(api_key=TOGETHER_API_KEY)

    prompt = f"""
    Analyze the following notes and create a detailed knowledge graph with key concepts and their relationships. 
    Focus on extracting the most important entities, their attributes, and the connections between them.
    
    Notes: {input}

    Generate a JSON object with the following structure:
    {{
        "nodes": [
            {{"id": "unique_id_1", "label": "Entity 1", "type": "person/concept/action/etc"}},
            {{"id": "unique_id_2", "label": "Entity 2", "type": "person/concept/action/etc"}},
            ...
        ],
        "edges": [
            {{"source": "unique_id_1", "target": "unique_id_2", "label": "relationship description"}},
            ...
        ]
    }}

    Ensure that:
    1. Each node has a unique id.
    2. Edge source and target ids correspond to existing node ids.
    3. Node labels are concise but descriptive.
    4. Edge labels clearly describe the relationship between nodes.
    5. Include at least 10 nodes and 15 edges, but not more than 30 nodes and 50 edges.

    Only respond with the JSON object, no additional text.
    """

    extract = together.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        response_format={
            "type": "json_object",
            "schema": KnowledgeGraph.model_json_schema(),
        },
    )

    try:
        output = json.loads(extract.choices[0].message.content)
        # Ensure the output matches our KnowledgeGraph structure
        graph = KnowledgeGraph(**output)
        return graph
    except json.JSONDecodeError:
        return KnowledgeGraph(nodes=[], edges=[])
    except ValueError:
        # If the structure doesn't match, return an empty graph
        return KnowledgeGraph(nodes=[], edges=[])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        notes = request.form['notes']
        graph = generate_graph(notes)
        return jsonify(graph.model_dump())
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
