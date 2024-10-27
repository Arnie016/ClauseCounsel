import streamlit as st
from test import get_response
import uuid
import re
import time
import random
import openai
import os
from dotenv import load_dotenv
import networkx as nx
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import json
from datetime import datetime
import pandas as pd
import altair as alt

load_dotenv()  # Load environment variables

# Initialize OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'saved_conversations' not in st.session_state:
    st.session_state.saved_conversations = []
if 'context' not in st.session_state:
    st.session_state.context = ""

def generate_unique_id():
    return str(uuid.uuid4())

def clean_response(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    return text


def add_to_conversation(question, response):
    cleaned_response = clean_response(response)
    st.session_state.conversation.append({
        'id': generate_unique_id(),
        'question': question,
        'response': cleaned_response,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def save_conversation(convo_id):
    for convo in st.session_state.conversation:
        if convo['id'] == convo_id and convo not in st.session_state.saved_conversations:
            st.session_state.saved_conversations.append(convo)
            return True
    return False

def delete_saved_conversation(convo_id):
    st.session_state.saved_conversations = [convo for convo in st.session_state.saved_conversations if convo['id'] != convo_id]

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def truncate_text(text, max_length=50):
    return text[:max_length] + '...' if len(text) > max_length else text

def create_card_web():
    net = Network(height='800px', width='100%', bgcolor='#222222', font_color='white', notebook=True)

    color_palette = ['#4287f5', '#f54242', '#42f554', '#f5a442', '#42f5f5', '#f542f5']

    for idx, convo in enumerate(st.session_state.conversation):
        color = random.choice(color_palette)
        tooltip = f"""Question: {truncate_text(convo['question'])}
Response: {truncate_text(convo['response'])}
Click to view full details"""
        net.add_node(idx, 
                     label=truncate_text(convo['question'], 30),
                     title=tooltip,
                     color=color,
                     size=30,
                     shape='dot',
                     font={'size': 14, 'face': 'Arial'},
                     borderWidth=2,
                     borderWidthSelected=4)

    for i in range(len(st.session_state.conversation)):
        for j in range(i+1, len(st.session_state.conversation)):
            similarity = calculate_similarity(
                st.session_state.conversation[i]['response'],
                st.session_state.conversation[j]['response']
            )
            if similarity > 0.3:
                net.add_edge(i, j, value=similarity, color='#ffffff', width=similarity * 3)

    net.set_options("""
    var options = {
      "nodes": {
        "shape": "dot",
        "scaling": {
          "min": 20,
          "max": 40
        },
        "font": {
          "size": 14,
          "face": "Arial"
        }
      },
      "edges": {
        "color": {"inherit": "both"},
        "smooth": {"type": "continuous"},
        "length": 250
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      },
      "interaction": {
        "tooltipDelay": 200,
        "hideEdgesOnDrag": true,
        "hover": true,
        "multiselect": true,
        "navigationButtons": true
      }
    }
    """)

    html_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    net.save_graph(html_file.name)
    
    with open(html_file.name, 'r', encoding='utf-8') as file:
        content = file.read()
    
    custom_css = """
    <style>
      body { background-color: #222222; }
      .vis-network { border: 1px solid #444444; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
      .vis-tooltip {
        background-color: #333333;
        border-color: #555555;
        border-radius: 4px;
        color: #ffffff;
        font-family: Arial, sans-serif;
        font-size: 14px;
        padding: 10px;
        max-width: 300px;
      }
      .vis-tooltip strong { color: #4287f5; }
    </style>
    """
    
    custom_js = """
    <script>
    network.on("click", function(params) {
        if (params.nodes.length > 0) {
            var nodeId = params.nodes[0];
            window.parent.postMessage({type: 'NODE_CLICK', nodeId: nodeId}, '*');
        }
    });
    </script>
    """
    
    content = content.replace('</head>', custom_css + '</head>')
    content = content.replace('</body>', custom_js + '</body>')
    
    with open(html_file.name, 'w', encoding='utf-8') as file:
        file.write(content)
    
    return html_file.name

def create_timeline():
    # Convert conversations to a DataFrame
    df = pd.DataFrame(st.session_state.conversation)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Create the chart
    chart = alt.Chart(df).mark_circle(size=100).encode(
        x='timestamp:T',
        y=alt.Y('index:O', axis=None),
        color=alt.Color('index:O', legend=None),
        tooltip=['question', 'response', 'timestamp']
    ).properties(
        width=800,
        height=400
    ).interactive()

    return chart

def main():
    st.set_page_config(page_title="ClauseCounsel", layout="wide")

    # Custom CSS for styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');
    
    body {
        font-family: 'Libre Baskerville', serif;
        color: #2c3e50;
        background-color: #ecf0f1;
        font-size: 18px;
    }
    .main-header {
        background-color: #34495e;
        color: #ecf0f1;
        padding: 40px;
        margin-bottom: 30px;
        text-align: center;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        font-size: 48px;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        font-size: 24px;
        opacity: 0.9;
    }
    .response-card {
        background-color: #f8f9fa;
        border-left: 5px solid #2980b9;
        padding: 25px;
        margin-bottom: 25px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: slideIn 0.5s ease-out;
        font-size: 20px;
    }
    .response-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        transform: translateY(-5px);
        background-color: #ffffff;
        border-left-color: #3498db;
    }
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .response-card strong {
        color: #2c3e50;
        font-size: 22px;
    }
    .inquiry {
        margin-bottom: 20px;
        padding-bottom: 20px;
        border-bottom: 1px solid #bdc3c7;
        color: #34495e;
    }
    .opinion {
        color: #2c3e50;
        line-height: 1.7;
    }
    .related-question {
        background-color: #e8f0fe;
        border: 1px solid #4285f4;
        border-radius: 4px;
        padding: 10px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .related-question:hover {
        background-color: #4285f4;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar for conversation history and search
    st.sidebar.title("Saved Conversations")
    search_query = st.sidebar.text_input("Search saved conversations:", "")
    
    filtered_conversations = st.session_state.saved_conversations
    if search_query:
        filtered_conversations = [
            convo for convo in st.session_state.saved_conversations
            if search_query.lower() in convo['question'].lower() or search_query.lower() in convo['response'].lower()
        ]

    for idx, saved_convo in enumerate(filtered_conversations):
        with st.sidebar.expander(f"Q: {truncate_text(saved_convo['question'])}", expanded=False):
            st.write(f"**Full Question:** {saved_convo['question']}")
            st.write(f"**Answer:** {truncate_text(saved_convo['response'], 100)}")
            col1, col2 = st.sidebar.columns([1, 1])
            with col1:
                if st.button("Load", key=f"load_{saved_convo['id']}"):
                    st.session_state.conversation = [saved_convo]
                    st.rerun()
            with col2:
                if st.button("Delete", key=f"delete_{saved_convo['id']}"):
                    delete_saved_conversation(saved_convo['id'])
                    st.rerun()

    # Main content
    st.markdown("""
    <div class="main-header">
        <h1>ClauseCounsel</h1>
        <p>Your AI Legal Assistant for Contract Disputes</p>
    </div>
    """, unsafe_allow_html=True)

    # Input field for the user's question
    question = st.text_input("Present your inquiry to the counsel:", "")

    # Create two columns for the buttons
    col1, col2, col3 = st.columns(3)

    # Submit button in the first column
    with col1:
        submit_button = st.button("Submit for Counsel's Opinion", key="submit_button")

    # Make a Card Web button in the second column
    with col2:
        web_button = st.button("Make a Card Web", key="web_button")

    # Add Timeline button next to Make a Card Web button
    with col3:
        timeline_button = st.button("Timeline")

    # Handle submission
    if submit_button and question:
        with st.spinner("Consulting legal databases..."):
            response = get_response(question)
        
        if isinstance(response, dict) and "message" in response:
            message = response["message"]
            
            if hasattr(message, 'text') and message.text:
                response_text = message.text
            elif hasattr(message, 'data') and 'text' in message.data:
                response_text = message.data['text']
            else:
                response_text = "No response text found."
            
            add_to_conversation(question, response_text)
            st.success("Counsel's opinion received")

            # Display the response
            st.markdown("### Counsel's Opinion")
            st.write(response_text)
        else:
            st.error("Unexpected response format")
            st.write(response)

    # Handle card web creation
    if web_button:
        if len(st.session_state.conversation) < 2:
            st.warning("Please add at least two cards to create a web.")
        else:
            with st.spinner("Creating Card Web..."):
                html_file = create_card_web()
                st.components.v1.html(open(html_file, 'r', encoding='utf-8').read(), height=800, scrolling=True)
                
                st.markdown("""
                ### Legend and Guidance
                - **Nodes**: Colored dots represent individual conversations
                - **Edges**: White lines connect similar conversations (thicker lines indicate higher similarity)
                
                **How to use:**
                - Hover over nodes to see a brief summary of the conversation
                - Click on a node to view the full conversation card
                - Drag nodes to rearrange the layout
                - Zoom in/out using the mouse wheel or pinch gesture
                - Use the navigation buttons in the bottom-right corner for additional interactions
                - Multi-select nodes by holding Ctrl (Cmd on Mac) while clicking
                """)

    # Handle timeline creation
    if timeline_button:
        if len(st.session_state.conversation) < 2:
            st.warning("Please add at least two cards to create a timeline.")
        else:
            with st.spinner("Creating Timeline..."):
                timeline_chart = create_timeline()
                st.altair_chart(timeline_chart)
                
                st.markdown("""
                ### Timeline Guidance
                - Hover over points to see details
                - Drag to pan the view
                - Scroll to zoom in/out
                - Click and drag to select a time range
                """)

    # JavaScript to handle node click events
    st.markdown("""
    <script>
    window.addEventListener('message', function(event) {
        if (event.data.type === 'NODE_CLICK') {
            Streamlit.setComponentValue({
                nodeId: event.data.nodeId
            });
            const element = document.getElementById('card-' + event.data.nodeId);
            if (element) {
                element.scrollIntoView({ behavior: 'smooth' });
            }
        }
    }, false);
    </script>
    """, unsafe_allow_html=True)

    # Display conversation history as cards
    st.markdown("### Case Proceedings")
    for idx, convo in enumerate(reversed(st.session_state.conversation)):
        with st.container():
            st.markdown(f"""
            <div id="card-{len(st.session_state.conversation) - idx - 1}" class="response-card">
                <div class="inquiry">
                    <strong style="color: #2980b9;">Inquiry:</strong> {convo['question']}
                </div>
                <div class="opinion">
                    <strong style="color: #2980b9;">Counsel's Opinion:</strong> {convo['response']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("Save to History", key=f"save_{convo['id']}"):
            if save_conversation(convo['id']):
                st.success("Conversation saved to history")
                st.rerun()
            else:
                st.info("This conversation is already saved in history")

    # Handle node click events
    if 'nodeId' in st.session_state:
        st.markdown(f"""
        <script>
        document.getElementById('card-{st.session_state.nodeId}').scrollIntoView({{ behavior: 'smooth' }});
        </script>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
