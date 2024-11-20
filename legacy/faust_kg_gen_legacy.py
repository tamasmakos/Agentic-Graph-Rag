# %%
# %pip install neo4j kor langchain_groq langchain_core langchain_experimental yfiles_jupyter_graphs networkx matplotlib pyvis sentence_transformers graphdatascience langgraph

# %%
from langchain_groq import ChatGroq
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_experimental.graph_transformers import LLMGraphTransformer

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    max_bucket_size=500,
)

# Define the language model (LLM) once
llm = ChatGroq(
    temperature=0.0,
    model_name="llama-3.2-11b-text-preview",
    rate_limiter=rate_limiter,
    api_key="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"
)

# Initialize the LLMGraphTransformer
llm_transformer = LLMGraphTransformer(
    llm=llm,
    strict_mode=True,
    node_properties=True,
    relationship_properties=True,
    ignore_tool_usage=False
)


# %%
import json
import re
import io
import pandas as pd
from kor import create_extraction_chain, Object, Text
from langchain_core.documents import Document


# Define the dramatic structure schema using KOR
dramatic_structure_schema = Object(
    id="scene_classification",
    description="Classify a scene from Faust according to its dramatic structure",
    attributes=[
        Text(
            id="dramatic_element",
            description="The dramatic element that best describes this scene",
            options=["Exposition", "Rising Action", "Climax", "Falling Action", "Resolution", "Structural Elements in Faust"]
        ),
        Text(
            id="justification",
            description="A brief explanation of why this scene fits the chosen dramatic element"
        ),
        Text(
            id="scene_description",
            description="A brief description of what is happening in the scene based on the eventstream"
        ),
        Text(
            id="scene_title",
            description="A suitable title for the scene based on the eventstream"
        ),
        Text(
            id="scene_type",
            description="The type of scene based on the eventstream"
        ),
        Text(
            id="scene_knowledge_graph",
            description="A knowledge graph representation of the scene based on the eventstream"
        )
    ]
)

# Create the extraction chain
classification_chain = create_extraction_chain(llm, dramatic_structure_schema)

# Define the dramatic structure dictionary
dramatic_structure_dict = {
        "Prologue": [
        "Dedication: Goethe's reflections on writing the play",
        "Prelude on the Stage: Discussion between Director, Poet, and Comedian about the nature of drama",
        "Prologue in Heaven: God's wager with Mephistopheles regarding Faust's soul"
        ],
        "Exposition": [
        "Introduction of Faust as a dissatisfied scholar in his study",
        "Faust's attempted suicide and the Easter chorus that saves him",
        "Faust's walk with Wagner and encounter with the black poodle",
        "Faust's first encounter with Mephistopheles in his study"
        ],
        "Rising Action": [
        "Faust's pact with Mephistopheles, signed in blood",
        "Faust's rejuvenation in the witch's kitchen",
        "Introduction of Gretchen (Margarete) and the beginning of their romance",
        "Faust and Mephistopheles' adventures in Auerbach's Cellar",
        "The gift of jewels to Gretchen and her growing attraction to Faust",
        "Faust's seduction of Gretchen, aided by Mephistopheles and Martha"
        ],
        "Climax": [
        "The death of Gretchen's mother due to the sleeping potion",
        "The duel where Faust kills Gretchen's brother, Valentin",
        "Gretchen's pregnancy and social ostracism",
        "The Walpurgis Night scene, contrasting with Gretchen's plight",
        "Faust's realization of Gretchen's imprisonment"
        ],
        "Falling Action": [
        "Faust's guilt and desperate attempts to save Gretchen",
        "The prison scene where Faust tries to convince Gretchen to escape"
        ],
        "Resolution": [
        "Gretchen's refusal to escape and her execution",
        "Gretchen's ascension to heaven and redemption",
        "Mephistopheles' declaration that Faust is still bound to him"
        ],
        "Structural Elements in Faust": [
        "Framing Device: The play begins and ends with heavenly scenes, emphasizing the cosmic nature of the struggle",
        "Episodic Structure: The play is composed of loosely connected scenes, allowing for a wide range of experiences and locations",
        "Parallel Plots: The cosmic struggle between good and evil is mirrored in Faust's personal journey and the tragedy of Gretchen",
        "Use of Verse and Prose: Goethe alternates between poetic verse and prose to differentiate between elevated and mundane scenes",
        "Symbolic Characters: Many characters represent broader concepts (e.g., Mephistopheles as temptation, Gretchen as innocence)",
        "Intertextuality: References to classical mythology, biblical stories, and folk tales enrich the narrative",
        "Contrast: Juxtaposition of the sublime and the grotesque, the spiritual and the earthly"
        ],
        "Themes Developed Through Structure": [
        "The limits of human knowledge and ambition",
        "The conflict between good and evil within the human soul",
        "The consequences of unchecked desire and ambition",
        "Redemption through love and divine grace",
        "The tension between medieval values and Enlightenment ideals"
        ]
}

def roman_to_int(roman):
    roman_dict = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
                  'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19,
                  'XX': 20}
    return roman_dict.get(roman.upper(), 0)

def roman_to_decimal(roman):
    # Convert a Roman numeral to a decimal for scene numbering
    int_value = roman_to_int(roman)
    return int_value

def incremental_classify_scene(scene, dramatic_structure_dict, previous_scenes):
    # Combine all events in the scene into a single text
    scene_text = "\n".join([json.dumps(event) for event in scene['events']])

    # Prepare the context from previous scenes
    if previous_scenes and len(previous_scenes) > 0:
        context = "\n\n".join([
            f"Scene {s.get('scene_node_id', 'Unknown')}: {s.get('dramatic_element', 'Unknown')}"
            for s in previous_scenes[-3:] if s is not None
        ])
    else:
        context = "No previous scenes"

    # Prepare the list of dramatic elements
    dramatic_elements_list = "\n".join(["- " + key for key in dramatic_structure_dict.keys()])

    # Prepare the input for the classification chain
    input_text = f"""
                    Previous context:
                    {context}

                    Classify the following scene from Faust according to its dramatic structure:

                    Scene: {scene['scene_node_id']}
                    Events:
                    {scene_text}

                    Dramatic Structure Elements:
                    {dramatic_elements_list}

                    Classify this scene into one of the dramatic structure elements and provide a detailed justification.
                    Consider the context of previous scenes when making your classification.
                    Provide the following:
                    - Dramatic Element
                    - Justification
                    - A brief description of what is happening in the scene based on the eventstream
                    - A suitable title for the scene based on the eventstream
                    - The type of scene based on the eventstream (e.g., dialogue, monologue, action)
                    Always provide the classification and additional information, even if you're not entirely certain.
                    """

    try:
        # Use invoke to get the output
        output = classification_chain.invoke({"text": input_text})
        parsed_output = output['data']
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        parsed_output = {}

    # Extract the classification result
    if parsed_output and 'scene_classification' in parsed_output and parsed_output['scene_classification']:
        classification = parsed_output['scene_classification']
        if isinstance(classification, list) and len(classification) > 0:
            classification = classification[0]
        scene['dramatic_element'] = classification.get('dramatic_element', 'Unclassified')
        scene['justification'] = classification.get('justification', 'No justification provided')
        scene['scene_description'] = classification.get('scene_description', 'No description provided')
        scene['scene_title'] = classification.get('scene_title', 'No title provided')
        scene['scene_type'] = classification.get('scene_type', 'Unknown')
    else:
        scene['dramatic_element'] = "Unclassified"
        scene['justification'] = "Classification failed or returned empty result"
        scene['scene_description'] = "No description provided."
        scene['scene_title'] = "No title provided."
        scene['scene_type'] = "Unknown"

    try:
        # Prepare the text from eventstream
        eventstream_text = "\n".join([f"{event.get('character', '')} {event.get('dialogue', '')}" for event in scene['events']])
        documents = [Document(page_content=eventstream_text)]
        graph_documents = llm_transformer.convert_to_graph_documents(documents)

        # Access the nodes and relationships directly from the GraphDocument
        nodes = graph_documents[0].nodes
        relationships = graph_documents[0].relationships

        # Convert nodes and relationships to a dictionary representation
        graph_dict = {
            "nodes": [{"id": make_serializable(node.id),
                    "type": make_serializable(node.type),
                    "properties": make_serializable(node.properties)} for node in nodes],
            "links": [{"source": make_serializable(rel.source.id),
                    "target": make_serializable(rel.target.id),
                    "type": make_serializable(rel.type),
                    "properties": make_serializable(rel.properties)} for rel in relationships]
        }
        scene['scene_knowledge_graph'] = graph_dict

    except Exception as e:
        print(f"Error during knowledge graph generation: {str(e)}")
        scene['scene_knowledge_graph'] = {"error": f"Knowledge graph generation failed: {str(e)}"}

    return scene


def parse_faust_text(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    lines = text.splitlines()
    lines = lines[0:5000]
    
    structured_data = []
    act = None
    scene = None
    previous_scenes = []

    i = 0

    while i < len(lines):
        line = lines[i].rstrip('\n')
        line_stripped = line.lstrip()

        # Detect Act
        act_match = re.match(r'^ACT\s*([IVX]+)\.?\s*$', line_stripped, re.IGNORECASE)
        if act_match:
            # Classify the previous scene if it exists
            if scene and scene['events']:
                scene = incremental_classify_scene(scene, dramatic_structure_dict, previous_scenes)
                previous_scenes.append(scene)
                act['scenes'].append(scene)
            act_number = act_match.group(1)
            act_sequence_number = roman_to_int(act_number)
            act = {'act_node_id': act_number, 'act_sequence_number': act_sequence_number, 'scenes': []}
            structured_data.append(act)
            scene = None
            i += 1
            continue

        # Detect Scene
        scene_match = re.match(r'^Scene ([IVX]+)\.$', line_stripped, re.IGNORECASE)
        if scene_match:
            # Classify the previous scene if it exists
            if scene and scene['events']:
                scene = incremental_classify_scene(scene, dramatic_structure_dict, previous_scenes)
                previous_scenes.append(scene)
                if act:
                    act['scenes'].append(scene)
            scene_number = scene_match.group(1)
            if act is None:
                act = {'act_node_id': 'Unknown', 'act_sequence_number': 0, 'scenes': []}
                structured_data.append(act)
            scene_sequence_number = act['act_sequence_number'] + roman_to_decimal(scene_number)
            scene = {
                'scene_node_id': scene_number,
                'scene_sequence_number': scene_sequence_number,
                'events': []
            }
            i += 1
            continue

        # Handle Empty Line
        if line.strip() == '':
            i += 1
            continue

        # Detect Character Name with possible stage direction
        char_match = re.match(r'^\s([^\s_].*?)\.(\s*_\w.*?_)?$', line)
        if char_match:
            character = char_match.group(1).strip().replace('.', '')
            act_description = char_match.group(2).replace('[_', '').replace('_]', '').strip() if char_match.group(2) else ''
            i += 1
            dialogue = []
            while i < len(lines) and lines[i].strip() != '' and lines[i].startswith(' '):
                dialogue.append(lines[i].strip())
                i += 1
            dialogue = ' '.join(dialogue)
            event_sequence_number = generate_event_sequence_number(act['act_sequence_number'], scene['scene_node_id'], len(scene['events']) + 1)
            event = {
                'character': character + '.',
                'dialogue': dialogue,
                'event_sequence_number': event_sequence_number
            }
            if act_description:
                event['action'] = act_description
            if scene is None:
                scene = {
                    'scene_node_id': 'Unknown',
                    'scene_sequence_number': 0.0,
                    'events': []
                }
                if act is None:
                    act = {'act_node_id': 'Unknown', 'act_sequence_number': 0, 'scenes': []}
                    structured_data.append(act)
                act['scenes'].append(scene)
            scene['events'].append(event)
            continue

        # Detect Stage Direction outside character's dialogue
        stage_direction_match = re.match(r'^\[(.*)\]$', line.strip())
        if stage_direction_match:
            stage_direction = stage_direction_match.group(1).strip()
            event_sequence_number = generate_event_sequence_number(act['act_sequence_number'], scene['scene_node_id'], len(scene['events']) + 1)
            event = {
                'stage_direction': stage_direction,
                'event_sequence_number': event_sequence_number
            }
            if scene is None:
                scene = {
                    'scene_node_id': 'Unknown',
                    'scene_sequence_number': 0.0,
                    'events': []
                }
                if act is None:
                    act = {'act_node_id': 'Unknown', 'act_sequence_number': 0, 'scenes': []}
                    structured_data.append(act)
                act['scenes'].append(scene)
            scene['events'].append(event)
            i += 1
            continue

        # Move to next line if none of the above matched
        i += 1

# Classify the last scene if it exists
    if scene and scene['events']:
        scene = incremental_classify_scene(scene, dramatic_structure_dict, previous_scenes)
        previous_scenes.append(scene)
        act['scenes'].append(scene)

    return json.dumps(structured_data, indent=2, ensure_ascii=False)


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif hasattr(obj, '__dict'):
        return make_serializable(obj.__dict__)
    else:
        return str(obj)


def generate_event_sequence_number(act_seq_num, scene_id, event_index):
    scene_decimal = roman_to_decimal(scene_id)
    return f"{act_seq_num}.{int(scene_decimal)}.{event_index}"


def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)
    return chunks

def build_graph_dataframes(structured_data_json):
    structured_data = json.loads(structured_data_json)
    nodes = []
    edges = []
    kg_nodes = []
    kg_edges = []

    previous_act_id = None

    for act_index, act in enumerate(structured_data):
        act_id = f"Act_{act_index+1}"
        act_label = 'Act'
        act_properties = {
            'act_node_id': act.get('act_node_id', ''),
            'act_sequence_number': act.get('act_sequence_number', '')
        }
        nodes.append({
            'id': act_id,
            'label': act_label,
            'properties': act_properties
        })

        # NEXT relationship between acts
        if previous_act_id is not None:
            edges.append({
                'source': previous_act_id,
                'target': act_id,
                'type': 'NEXT',
                'properties': {}
            })
        previous_act_id = act_id

        previous_scene_id = None
        scenes = act.get('scenes', [])
        for scene_index, scene in enumerate(scenes):
            scene_id = f"{act_id}_Scene_{scene_index+1}"
            scene_label = 'Scene'
            scene_properties = {
                'scene_node_id': scene.get('scene_node_id', ''),
                'scene_sequence_number': scene.get('scene_sequence_number', ''),
                'dramatic_element': scene.get('dramatic_element', ''),
                'justification': scene.get('justification', ''),
                'scene_description': scene.get('scene_description', ''),
                'scene_title': scene.get('scene_title', ''),
                'scene_type': scene.get('scene_type', '')
            }
            nodes.append({
                'id': scene_id,
                'label': scene_label,
                'properties': scene_properties
            })

            # PART_OF relationship between scene and act
            edges.append({
                'source': scene_id,
                'target': act_id,
                'type': 'PART_OF',
                'properties': {}
            })

            # NEXT relationship between scenes
            if previous_scene_id is not None:
                edges.append({
                    'source': previous_scene_id,
                    'target': scene_id,
                    'type': 'NEXT',
                    'properties': {}
                })
            previous_scene_id = scene_id

            previous_event_id = None
            events = scene.get('events', [])
            for event_index, event in enumerate(events):
                event_id = f"{scene_id}_Event_{event_index+1}"
                event_label = 'Event'
                event_properties = event.copy()
                nodes.append({
                    'id': event_id,
                    'label': event_label,
                    'properties': event_properties
                })

                # PART_OF relationship between event and scene
                edges.append({
                    'source': event_id,
                    'target': scene_id,
                    'type': 'PART_OF',
                    'properties': {}
                })

                # NEXT relationship between events
                if previous_event_id is not None:
                    edges.append({
                        'source': previous_event_id,
                        'target': event_id,
                        'type': 'NEXT',
                        'properties': {}
                    })
                previous_event_id = event_id

            # Now process chunks and knowledge graph
            eventstream_text = "\n".join([f"{event.get('character', '')} {event.get('dialogue', '')}" for event in scene['events']])
            chunks = split_text_into_chunks(eventstream_text, chunk_size=500)
            previous_chunk_id = None
            for chunk_index, chunk_text in enumerate(chunks):
                chunk_id = f"{scene_id}_Chunk_{chunk_index+1}"
                chunk_label = 'Chunk'
                chunk_properties = {
                    'text': chunk_text
                }
                nodes.append({
                    'id': chunk_id,
                    'label': chunk_label,
                    'properties': chunk_properties
                })

                # PART_OF relationship between chunk and scene
                edges.append({
                    'source': chunk_id,
                    'target': scene_id,
                    'type': 'PART_OF',
                    'properties': {}
                })

                # NEXT relationship between chunks
                if previous_chunk_id is not None:
                    edges.append({
                        'source': previous_chunk_id,
                        'target': chunk_id,
                        'type': 'NEXT',
                        'properties': {}
                    })
                previous_chunk_id = chunk_id

                # Run LLMGraphTransformer on the chunk
                try:
                    documents = [Document(page_content=chunk_text)]
                    graph_documents = llm_transformer.convert_to_graph_documents(documents)

                    # Extract nodes and relationships from graph_documents
                    for graph_doc in graph_documents:
                        for kg_node in graph_doc.nodes:
                            kg_node_id = f"{chunk_id}_{kg_node.id}"
                            kg_node_label = kg_node.type
                            kg_node_properties = kg_node.properties.copy()  # Create a copy of the properties
                            kg_node_properties['chunk_id'] = chunk_id
                            kg_node_properties['scene_id'] = scene_id
                            kg_node_properties['entity'] = kg_node.id  # Store the original entity name
                            kg_nodes.append({
                                'id': kg_node_id,
                                'entity': kg_node.id,
                                'label': kg_node_label,
                                'properties': kg_node_properties
                            })

                            # HAS_ENTITY relationship between chunk and entity
                            kg_edges.append({
                                'source': chunk_id,
                                'target': kg_node_id,
                                'type': 'HAS_ENTITY',
                                'properties': {}
                            })
                        for kg_rel in graph_doc.relationships:
                            kg_edge_source = f"{chunk_id}_{kg_rel.source.id}"
                            kg_edge_target = f"{chunk_id}_{kg_rel.target.id}"
                            kg_edges.append({
                                'source': kg_edge_source,
                                'target': kg_edge_target,
                                'type': kg_rel.type,
                                'properties': kg_rel.properties
                            })
                except Exception as e:
                    print(f"Error during knowledge graph generation for chunk: {str(e)}")

    # Create DataFrames
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    kg_nodes_df = pd.DataFrame(kg_nodes)
    kg_edges_df = pd.DataFrame(kg_edges)

    return nodes_df, edges_df, kg_nodes_df, kg_edges_df

import pandas as pd

def expand_properties(df):
    # Extract the properties column
    properties_df = pd.json_normalize(df['properties'])
    
    # Drop the original properties column
    df = df.drop('properties', axis=1)
    
    # Concatenate the original DataFrame with the expanded properties
    return pd.concat([df, properties_df], axis=1)

# Now, parse the text and build the DataFrames
structured_data_json = parse_faust_text('faust.txt')
nodes_df, edges_df, kg_nodes_df, kg_edges_df = build_graph_dataframes(structured_data_json)

# Expand properties for each DataFrame
nodes_df = expand_properties(nodes_df)
edges_df = expand_properties(edges_df)
kg_nodes_df = expand_properties(kg_nodes_df)
kg_edges_df = expand_properties(kg_edges_df)

# Save the DataFrames to CSV or any preferred format
nodes_df.to_csv('nodes.csv', index=False)
edges_df.to_csv('edges.csv', index=False)
kg_nodes_df.to_csv('kg_nodes.csv', index=False)
kg_edges_df.to_csv('kg_edges.csv', index=False)

# %%
from pyvis.network import Network
import pandas as pd

def create_pyvis_network(nodes_df, edges_df, kg_nodes_df, kg_edges_df):
    # Create a Pyvis network
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

    # Add nodes
    for _, row in nodes_df.iterrows():
        node_properties = row.dropna().to_dict()
        node_id = node_properties.pop('id')
        node_label = node_properties.pop('label', 'Unknown')

        # For Events, include the character value in the label
        if node_label == 'Event' and 'character' in node_properties:
            node_label = f"Event: {node_properties['character']}"

        try:
            net.add_node(node_id, label=node_label, title=str(node_properties), color=get_color(node_label))
        except:
            print(f"Error adding node: {node_id}")

    # Add edges
    for _, row in edges_df.iterrows():
        source = row['source']
        target = row['target']
        edge_type = row['type']

        # Add the relationship type as a label on the edge
        try:
            net.add_edge(source, target, title=edge_type, label=edge_type, font={'size': 6})
        except:
            print(f"Error adding edge between {source} and {target}")

    # Add knowledge graph nodes
    for _, row in kg_nodes_df.iterrows():
        kg_node_properties = row.dropna().to_dict()
        kg_node_id = kg_node_properties.pop('id')
        kg_node_label = kg_node_properties.pop('label', 'KG_Unknown')

        # Include the entity value in the label for knowledge graph nodes
        if 'entity' in kg_node_properties:
            kg_node_label = f"{kg_node_label}: {kg_node_properties['entity']}"

        try:
            net.add_node(kg_node_id, label=kg_node_label, title=str(kg_node_properties), color=get_color(kg_node_label))
        except:
            print(f"Error adding kg node: {kg_node_id}")

    # Add knowledge graph edges
    for _, row in kg_edges_df.iterrows():
        kg_source = row['source']
        kg_target = row['target']
        kg_edge_type = row['type']

        # Add the relationship type as a label on the edge
        try:
            net.add_edge(kg_source, kg_target, title=kg_edge_type, label=kg_edge_type, font={'size': 6})
        except:
            print(f"Error adding kg edge between {kg_source} and {kg_target}")

    # Connect knowledge graph nodes to chunk nodes
    for _, row in kg_nodes_df.iterrows():
        kg_node_id = row['id']
        chunk_id = row['chunk_id']

        if chunk_id in net.get_nodes():
            try:
                net.add_edge(chunk_id, kg_node_id, title='HAS_ENTITY', label='HAS_ENTITY', font={'size': 6})
            except:
                print(f"Error connecting kg node {kg_node_id} to chunk node {chunk_id}")

    # Set global options for edge labels
    net.set_edge_smooth('dynamic')  # This can help with label visibility

    return net

def get_color(node_type):
    color_map = {
        'Act': '#FF9999',
        'Scene': '#66B2FF',
        'Event': '#99FF99',
        'Chunk': '#FFCC99',
        'Entity': '#FF99FF'
    }
    return color_map.get(node_type, '#FFFFFF')

# Create and visualize the network
net = create_pyvis_network(nodes_df, edges_df, kg_nodes_df, kg_edges_df)

# Set some display options
net.toggle_physics(True)
net.show_buttons(filter_=['physics'])

# Save and show the network
net.save_graph("faust_network.html")

# %%
print('go for it')

# %% [markdown]
# #### NEO4J GDS

# %%
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience

### User
NEO4J_USERNAME = "neo4j"
NEO4J_DATABASE = "neo4j"

### GDB Insatnce
# Neo4j connection setup
# NEO4J_URI = "neo4j+ssc://f3e13232.databases.neo4j.io"
# NEO4J_PASSWORD = "7XIfeMdlGqzbzpBOCoePwWMSJltaHP4L598VjqjwXbE"

### GDS Instance
NEO4J_URI = "neo4j+s://db6a9f9c.databases.neo4j.io"
NEO4J_PASSWORD = "5G3vK13O9qdE01F5R270JzldVIzYThrCxeVxJaMFf4c"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
gds = GraphDataScience(NEO4J_URI, auth=("neo4j", NEO4J_PASSWORD))

# %%
import re
from neo4j.exceptions import CypherSyntaxError
import pandas as pd

def upload_to_neo4j(nodes_df, edges_df, kg_nodes_df, kg_edges_df):
    kg_nodes_df = kg_nodes_df.loc[:, ~kg_nodes_df.columns.duplicated()]
    kg_edges_df = kg_edges_df.loc[:, ~kg_edges_df.columns.duplicated()]
    
    with driver.session(database=NEO4J_DATABASE) as session:
        try:
            session.run('MATCH (n) DETACH DELETE n')
        except Exception as e:
            print(f"Error clearing database: {str(e)}")

        # Upload non-KG nodes
        for _, row in nodes_df.iterrows():
            try:
                label = re.sub(r'\W+', '_', row['label'])
                node_id = row['id']
                properties = {k: v for k, v in row.items() if k not in ['id', 'label'] and pd.notna(v)}
                cypher_query = (
                    f"MERGE (n:`{label}` {{id: $node_id}}) "
                    "SET n += $properties "
                    "RETURN n"
                )
                session.run(cypher_query, node_id=node_id, properties=properties)
            except CypherSyntaxError as e:
                print(f"Cypher syntax error for node {node_id}: {str(e)}")
            except Exception as e:
                print(f"Error uploading node {node_id}: {str(e)}")

        # Upload KG nodes
        for _, row in kg_nodes_df.iterrows():
            try:
                label = row['label']
                entity_name = row['entity'][0] if isinstance(row['entity'], list) else row['entity']
                
                # First MERGE the base Entity node
                base_query = (
                    "MERGE (n:Entity {name: $entity_name}) "
                    "RETURN n"
                )
                session.run(base_query, entity_name=entity_name)
                
                # Then add the additional label
                label_query = (
                    "MATCH (n:Entity {name: $entity_name}) "
                    f"SET n:`{label}` "
                    "RETURN n"
                )
                session.run(label_query, entity_name=entity_name)
                
            except CypherSyntaxError as e:
                print(f"Cypher syntax error for KG node {entity_name}: {str(e)}")
            except Exception as e:
                print(f"Error uploading KG node {entity_name}: {str(e)}")

        # Upload non-KG edges
        for _, row in edges_df.iterrows():
            try:
                properties = {k: v for k, v in row.items() if k not in ['source', 'target', 'type'] and pd.notna(v)}
                relationship_type = re.sub(r'\W+', '_', row['type'])
                source_id = row['source']
                target_id = row['target']
                cypher_query = (
                    "MATCH (source {id: $source_id}) "
                    "MATCH (target {id: $target_id}) "
                    f"MERGE (source)-[r:`{relationship_type}`]->(target) "
                    "SET r += $properties "
                    "RETURN r"
                )
                session.run(cypher_query, source_id=source_id, target_id=target_id, properties=properties)
            except CypherSyntaxError as e:
                print(f"Cypher syntax error for edge {source_id} -> {target_id}: {str(e)}")
            except Exception as e:
                print(f"Error uploading edge {source_id} -> {target_id}: {str(e)}")

        # Upload KG edges
        for _, row in kg_edges_df.iterrows():
            try:
                properties = {k: v for k, v in row.items() if k not in ['source', 'target', 'type'] and pd.notna(v)}
                relationship_type = re.sub(r'\W+', '_', row['type'])
                source_name = row['source'].split('_')[-1] if isinstance(row['source'], str) else row['source'][0]
                target_name = row['target'].split('_')[-1] if isinstance(row['target'], str) else row['target'][0]
                cypher_query = (
                    "MATCH (source:Entity {name: $source_name}) "
                    "MATCH (target:Entity {name: $target_name}) "
                    f"MERGE (source)-[r:`{relationship_type}`]->(target) "
                    "SET r += $properties "
                    "RETURN r"
                )
                session.run(cypher_query, source_name=source_name, target_name=target_name, properties=properties)
            except CypherSyntaxError as e:
                print(f"Cypher syntax error for KG edge {source_name} -> {target_name}: {str(e)}")
            except Exception as e:
                print(f"Error uploading KG edge {source_name} -> {target_name}: {str(e)}")

        # Create HAS_ENTITY connections
        for _, row in kg_nodes_df.iterrows():
            try:
                chunk_id = row['chunk_id']
                entity_name = row['entity'][0] if isinstance(row['entity'], list) else row['entity']
                node_id = row['id']
                cypher_query = (
                    "MATCH (chunk {id: $chunk_id}) "
                    "MATCH (entity:Entity {name: $entity_name}) "
                    "MERGE (chunk)-[r:HAS_ENTITY]->(entity) "
                    "SET r.id = $node_id "
                    "RETURN r"
                )
                session.run(cypher_query, chunk_id=chunk_id, entity_name=entity_name, node_id=node_id)
            except CypherSyntaxError as e:
                print(f"Cypher syntax error for HAS_ENTITY connection {chunk_id} -> {entity_name}: {str(e)}")
            except Exception as e:
                print(f"Error creating HAS_ENTITY connection {chunk_id} -> {entity_name}: {str(e)}")
        

        # Create HAS_DIALOGUE connections between Event and Entity nodes
        try:
            cypher_query = """
            MATCH (e:Event)
            WHERE e.character IS NOT NULL
            WITH e, trim(replace(e.character, '.', '')) AS trimmed_character
            MATCH (entity:Entity)
            WHERE entity.name = trimmed_character
            MERGE (e)-[r:HAS_DIALOGUE]->(entity)
            RETURN count(r) as created_relationships
            """
            result = session.run(cypher_query)
            created_relationships = result.single()['created_relationships']
            print(f"Created {created_relationships} HAS_DIALOGUE relationships")
        except CypherSyntaxError as e:
            print(f"Cypher syntax error for HAS_DIALOGUE connections: {str(e)}")
        except Exception as e:
            print(f"Error creating HAS_DIALOGUE connections: {str(e)}")

        try:
            cypher_query = """
            MATCH (e:Entity)
            WITH e.name AS name, collect(e) AS nodes
            WHERE size(nodes) > 1
            CALL apoc.refactor.mergeNodes(nodes, {properties:"combine", mergeRels:true})
            YIELD node
            RETURN node
            """
            session.run(cypher_query)
        except CypherSyntaxError as e:
            print(f"Cypher syntax error for merging Entity nodes: {str(e)}")
        except Exception as e:
            print(f"Error merging Entity nodes: {str(e)}")
            

    print("Knowledge graph upload to Neo4j completed!")

nodes_df = pd.read_csv('nodes.csv')
edges_df = pd.read_csv('edges.csv')
kg_nodes_df = pd.read_csv('kg_nodes.csv')
kg_edges_df = pd.read_csv('kg_edges.csv')

# Call the function once with all dataframes
upload_to_neo4j(nodes_df, edges_df, kg_nodes_df, kg_edges_df)

# %%
from yfiles_jupyter_graphs import GraphWidget

def showGraph():
    session = driver.session()
    w = GraphWidget(graph = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 1000").graph())
    def custom_node_label_mapping(node: dict):
        # print(f"Node data: {node}")  # Debug print
        properties = node.get('properties', {})
        label = properties.get('label', '')
        # print(f"Label: {label}, Properties: {properties}")  # Debug print
        
        if 'Entity' in label:
            return {properties.get('name', 'Unknown')}
        elif 'Event' in label:
            return {properties.get('character', 'Unknown')}
        elif 'Act' in label:
            return {properties.get('act_node_id', 'Unknown')}
        elif 'Scene' in label:
            return {properties.get('id', '').split('_')[-1]}
        elif 'Chunk' in label:
            return {properties.get('id', 'Unknown')}
        
        # Default case if no matching label
        return f"Unknown ({label})"
    w.set_node_label_mapping(custom_node_label_mapping)
    
    return w

showGraph()

# %%
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph(
    url = NEO4J_URI,
    username= NEO4J_USERNAME,
    password= NEO4J_PASSWORD,
    database= NEO4J_DATABASE
    )

exists_result = gds.graph.exists("communities")
if exists_result['exists'].item():  # Use item() instead of bool() for numpy.bool_
    gds.graph.drop("communities")

G, result = gds.graph.project(
    "communities",  #  Graph name
    "Entity",  #  Node projection
    {
        "_ALL_": {
            "type": "*",
            "orientation": "UNDIRECTED",
            "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
        }
    },
)

wcc = gds.wcc.stats(G)
print(f"Component count: {wcc['componentCount']}")
print(f"Component distribution: {wcc['componentDistribution']}")

gds.leiden.write(
    G,
    writeProperty="communities",
    includeIntermediateCommunities=True,
    relationshipWeightProperty="weight",
)

gds.graph.drop("communities")

exists_result = gds.graph.exists("centrality_graph")
if exists_result['exists'].item():
    gds.graph.drop("centrality_graph")


G, result = gds.graph.project(
    "centrality_graph",  #  Graph name
    "Entity",  #  Node projection
    {
        "_ALL_": {
            "type": "*",
            "orientation": "NATURAL",
            "properties": {
                "weight": {
                    "property": "*",  # Changed from "weight" to "*" to count parallel relationships
                    "aggregation": "COUNT"  # Changed from "SINGLE" to "COUNT"
                }
            }
        }
    }
)

# Calculate and write degree centrality
gds.degree.write(
    G,
    writeProperty="degree_centrality",
    relationshipWeightProperty="weight"
)

# Calculate and write betweenness centrality
gds.betweenness.write(
    G,
    writeProperty="betweenness_centrality"
)

# Calculate and write eigenvector centrality
gds.eigenvector.write(
    G,
    writeProperty="eigenvector_centrality",
    relationshipWeightProperty="weight",
    maxIterations=100
)

print("Centrality measures have been written to the graph")

# Optional: Drop the projected graph to free up memory
gds.graph.drop("centrality_graph")

# First, let's check if we have entities with communities property
result = graph.query("""
MATCH (e:Entity)
WHERE e.communities IS NOT NULL
RETURN count(e) as entity_count
""")
print(f"Entities with communities property: {result}")

# First, let's check a sample entity
result = graph.query("""
MATCH (e:Entity)
WHERE e.communities IS NOT NULL
RETURN e.name as name, e.communities as communities, e.degree_centrality, e.betweenness_centrality, e.eigenvector_centrality
LIMIT 1
""")
print("Sample entity communities:", result)

# Now let's check the counts step by step
result = graph.query("""
MATCH (e:Entity)
WHERE e.communities IS NOT NULL
WITH count(e) as initial_count
MATCH (e:Entity)
WHERE e.communities IS NOT NULL
UNWIND range(0, size(e.communities) - 1) AS index
WITH initial_count, count(*) as after_unwind
MATCH (e:Entity)
WHERE e.communities IS NOT NULL
UNWIND range(0, size(e.communities) - 1) AS index
WITH e, index, initial_count, after_unwind
MERGE (c:Community {id: toString(index) + '-' + toString(e.communities[index])})
WITH initial_count, after_unwind, count(*) as after_community_creation
RETURN initial_count, after_unwind, after_community_creation
""")
print("\nCounts at each step:", result)

# Check chunk connections
result = graph.query("""
MATCH (chunk:Chunk)-[:HAS_ENTITY]->(e:Entity)
WHERE e.communities IS NOT NULL
RETURN count(*) as chunk_entity_connections
""")
print("\nChunk-Entity connections:", result)


def create_community_structure(graph):
    result = graph.query("""
    // First create communities and connect entities
    MATCH (e:Entity)
    WHERE e.communities IS NOT NULL
    UNWIND e.communities AS community_id
    MERGE (c:Community {id: community_id})
    MERGE (e)-[:BELONGS_TO]->(c)
    
    // Connect chunks to communities 
    WITH DISTINCT e, c
    MATCH (chunk:Chunk)-[:HAS_ENTITY]->(e)
    MERGE (chunk)-[:ASSOCIATED_WITH]->(c)
    
    RETURN count(DISTINCT e) AS entities_processed,
           count(DISTINCT c) AS communities_created,
           count(DISTINCT chunk) AS chunks_associated
    """)
    
    print(f"Community structure created: {result}")

# Call the function
create_community_structure(graph)

# %%
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings


# Configuration constants
topChunks = 3
topCommunities = 3
topOutsideRels = 10
topInsideRels = 10
topEntities = 10

# Modified Event retrieval query
ev_retrieval_query = """
WITH collect(node) as nodes
// Event-Chunk Mapping
WITH nodes, collect {
    UNWIND nodes as n
    MATCH (n:Event)-[:HAS_ENTITY]->(e:Entity)<-[:HAS_ENTITY]-(c:Chunk)
    WITH c, count(distinct n) as freq, n
    // Add centrality-based ordering if specified
    ORDER BY 
        CASE 
            WHEN $centrality_field IS NOT NULL THEN n[$centrality_field]
            ELSE freq 
        END DESC,
        freq DESC
    LIMIT $topChunks
    RETURN c.text AS chunkText
} AS text_mapping,
// Event-Community Mapping
collect {
    UNWIND nodes as n
    MATCH (n:Event)-[:HAS_ENTITY]->(e:Entity)-[:BELONGS_TO]->(c:Community)
    WITH c, c.community_rank as rank, c.weight as weight
    RETURN 'Community ' + c.id + ' (rank: ' + toString(rank) + ', weight: ' + toString(weight) + ')' as summary
    ORDER BY rank DESC, weight DESC
    LIMIT $topCommunities
} AS community_mapping,
// Event-Entity Relationships
collect {
    UNWIND nodes as n
    MATCH (n:Event)-[r:HAS_DIALOGUE]->(e:Entity)
    WHERE NOT e IN nodes
    RETURN 'Character ' + e.name + ' has dialogue in event' as descriptionText
    LIMIT $topOutsideRels
} as dialogue_rels,
// Event-Event Relationships
collect {
    UNWIND nodes as n
    MATCH (n:Event)-[:HAS_ENTITY]->(e:Entity)<-[:HAS_ENTITY]-(m:Event)
    WHERE m IN nodes AND id(n) < id(m)
    RETURN 'Events share entity: ' + e.name as descriptionText
    LIMIT $topInsideRels
} as shared_entity_rels,
// Event descriptions
collect {
    UNWIND nodes as n
    RETURN n.dialogue as descriptionText
} as events
RETURN {
    Chunks: text_mapping,
    Communities: community_mapping,
    DialogueRelationships: dialogue_rels,
    SharedEntities: shared_entity_rels,
    Events: events
} AS text, 1.0 AS score, {} AS metadata
"""

# Modified Chunk retrieval query
cv_retrieval_query = """
WITH collect(node) as nodes
// Chunk-Entity Mapping
WITH nodes, collect {
    UNWIND nodes as n
    MATCH (n:Chunk)-[:HAS_ENTITY]->(e:Entity)
    WITH e, count(distinct n) as freq, e.weight as weight, n
    // Add centrality-based ordering if specified
    ORDER BY 
        CASE 
            WHEN $centrality_field IS NOT NULL THEN n[$centrality_field]
            ELSE freq 
        END DESC,
        freq DESC, 
        weight DESC
    LIMIT $topChunks
    RETURN e.name AS entityText
} AS entity_mapping,
// Chunk-Community Mapping
collect {
    UNWIND nodes as n
    MATCH (n:Chunk)-[:ASSOCIATED_WITH]->(c:Community)
    WITH c, c.community_rank as rank
    RETURN 'Community ' + c.id + ' (rank: ' + toString(rank) + ')' as summary
    ORDER BY rank DESC
    LIMIT $topCommunities
} AS community_mapping,
// Related Chunks
collect {
    UNWIND nodes as n
    MATCH (n:Chunk)-[:HAS_ENTITY]->(e:Entity)<-[:HAS_ENTITY]-(m:Chunk)
    WHERE NOT m IN nodes
    RETURN 'Related chunk shares entity: ' + e.name as descriptionText
    LIMIT $topOutsideRels
} as related_chunks,
// Community Chunks
collect {
    UNWIND nodes as n
    MATCH (n:Chunk)-[:ASSOCIATED_WITH]->(c:Community)<-[:ASSOCIATED_WITH]-(m:Chunk)
    WHERE m IN nodes AND id(n) < id(m)
    RETURN 'Chunks in community: ' + c.id as descriptionText
    LIMIT $topInsideRels
} as community_chunks,
// Chunk content
collect {
    UNWIND nodes as n
    RETURN n.text as descriptionText
} as chunks
RETURN {
    Entities: entity_mapping,
    Communities: community_mapping,
    RelatedChunks: related_chunks,
    CommunityChunks: community_chunks,
    Content: chunks
} AS text, 1.0 AS score, {} AS metadata
"""

class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# Use the wrapper class
embeddings = LocalHuggingFaceEmbeddings('all-MiniLM-L6-v2')

def calculate_community_weights(graph):
    result = graph.query("""
    MATCH (n:Community)<-[:BELONGS_TO]-(e:Entity)<-[:HAS_ENTITY]-(c:Chunk)
    WITH n, count(distinct c) AS chunkCount
    SET n.weight = chunkCount,
        // Add community rank if not exists
        n.community_rank = coalesce(n.community_rank, 0)
    RETURN count(n) as communities_weighted
    """)
    print(f"Community weights calculated for {result} communities")

def create_embeddings_and_store():
    with driver.session(database=NEO4J_DATABASE) as session:
        # Create vector index for Event nodes with configurable dimensions
        session.run("""
        CREATE VECTOR INDEX event_vector IF NOT EXISTS
        FOR (e:Event)
        ON e.embedding
        OPTIONS { indexConfig: {
            `vector.dimensions`: 384,  // Matches the all-MiniLM-L6-v2 model dimensions
            `vector.similarity_function`: 'cosine'
        }}
        """)

        # Create vector index for Chunk nodes
        session.run("""
        CREATE VECTOR INDEX chunk_vector IF NOT EXISTS
        FOR (c:Chunk)
        ON c.embedding
        OPTIONS { indexConfig: {
            `vector.dimensions`: 384,  // Matches the all-MiniLM-L6-v2 model dimensions
            `vector.similarity_function`: 'cosine'
        }}
        """)
        
        # Fetch all Event nodes with dialogue
        event_result = session.run("""
            MATCH (e:Event)
            WHERE e.dialogue IS NOT NULL
            RETURN e.id AS id, e.dialogue AS dialogue
        """)
        
        events = [(record["id"], record["dialogue"]) for record in event_result]

        # Fetch all Chunk nodes with text
        chunk_result = session.run("""
            MATCH (c:Chunk)
            WHERE c.text IS NOT NULL
            RETURN c.id AS id, c.text AS text
        """)
        
        chunks = [(record["id"], record["text"]) for record in chunk_result]

        # Create embeddings for Events
        batch_size = 20
        for i in range(0, len(events), batch_size):
            batch = events[i:i+batch_size]
            texts = [dialogue for _, dialogue in batch]
            
            # Generate embeddings
            embeddings_batch = embeddings.embed_documents(texts)
            
            # Store embeddings in Neo4j
            for (node_id, _), embedding in zip(batch, embeddings_batch):
                session.run("""
                    MATCH (e:Event {id: $node_id})
                    CALL db.create.setNodeVectorProperty(e, 'embedding', $embedding)
                    RETURN e
                """, node_id=node_id, embedding=embedding)

        # Create embeddings for Chunks
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [text for _, text in batch]
            
            # Generate embeddings
            embeddings_batch = embeddings.embed_documents(texts)
            
            # Store embeddings in Neo4j
            for (node_id, _), embedding in zip(batch, embeddings_batch):
                session.run("""
                    MATCH (c:Chunk {id: $node_id})
                    CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
                    RETURN c
                """, node_id=node_id, embedding=embedding)
        
        print(f"Embeddings created and stored for {len(events)} Event nodes and {len(chunks)} Chunk nodes.")

def create_neo4j_vectors():
    # Create Neo4jVector instances
    event_vector = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="event_vector",
        node_label="Event",
        text_node_property="dialogue",
        embedding_node_property="embedding",
        retrieval_query = ev_retrieval_query
    )

    chunk_vector = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="chunk_vector",
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding",
        retrieval_query = cv_retrieval_query
    )

    return event_vector, chunk_vector

def similarity_search(query, vector_store, k=3, order_by_centrality=None):
    """
    Perform similarity search with optional centrality-based ordering
    
    Args:
        query: Search query text
        vector_store: Neo4jVector instance
        k: Number of results to return
        order_by_centrality: One of ['degree', 'betweenness', 'eigenvector'] or None
    """
    # Base parameters that are always needed
    params = {
        "topChunks": topChunks,
        "topCommunities": topCommunities,
        "topOutsideRels": topOutsideRels,
        "topInsideRels": topInsideRels,
        "topEntities": topEntities,
        "centrality_field": None  # Default to None when not using centrality
    }
    
    # Modify centrality field if specified
    if order_by_centrality:
        centrality_field = {
            'degree': 'degree_centrality',
            'betweenness': 'betweenness_centrality',
            'eigenvector': 'eigenvector_centrality'
        }.get(order_by_centrality)
        
        if not centrality_field:
            raise ValueError("order_by_centrality must be one of: degree, betweenness, eigenvector")
        
        params["centrality_field"] = centrality_field

    results = vector_store.similarity_search(query, k=k, params=params)
    return results

if __name__ == "__main__":
    # Create embeddings and store them
    create_embeddings_and_store()
    
    # Create Neo4jVector instances
    event_vector, chunk_vector = create_neo4j_vectors()
    
    # Example similarity search for Event nodes
    event_query = "A character's dialogue about a mysterious event"
    event_results = similarity_search(event_query, event_vector)

    # Order by degree centrality
    results_degree = similarity_search(event_query, event_vector, k=3, order_by_centrality='degree')

    # Order by betweenness centrality
    results_betweenness = similarity_search(event_query, event_vector, k=3, order_by_centrality='betweenness')

    # Order by eigenvector centrality
    results_eigenvector = similarity_search(event_query, event_vector, k=3, order_by_centrality='eigenvector')

    # Default ordering (no centrality)
    results_default = similarity_search(event_query, event_vector, k=3)
    
    print(f"\nTop 3 similar Event nodes for query: '{event_query}'")
    for i, result in enumerate(event_results, 1):
        print(f"{i}. {result.page_content}")

    # Example similarity search for Chunk nodes
    chunk_query = "A character's dialogue about a mysterious event"
    chunk_results = similarity_search(chunk_query, chunk_vector)
    
    print(f"\nTop 3 similar Chunk nodes for query: '{chunk_query}'")
    for i, result in enumerate(chunk_results, 1):
        print(f"{i}. {result.page_content}")

driver.close()

### Retry if you can't connect

# %%
print(f"\nTop 3 similar Event nodes for query: '{event_query}'")
for i, result in enumerate(results_degree, 1):
    print(f"{i}. {result.page_content}")

print(f"\nTop 3 similar Chunk nodes for query: '{chunk_query}'")
for i, result in enumerate(results_betweenness, 1):
    print(f"{i}. {result.page_content}")

print(f"\nTop 3 similar Chunk nodes for query: '{chunk_query}'")
for i, result in enumerate(results_eigenvector, 1):
    print(f"{i}. {result.page_content}")

# %% [markdown]
# #### LANGGRAPH

# %%
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.tools import Tool
from typing import Annotated, Sequence, Literal, List, Tuple, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage


# Create retriever tool using your Neo4j vectors
def retrieve_from_neo4j(query: str) -> str:
    """Retrieve relevant information from Neo4j knowledge graph."""
    event_results = event_vector.similarity_search(query, k=3)
    chunk_results = chunk_vector.similarity_search(query, k=3)
    
    combined_results = []
    for doc in event_results + chunk_results:
        if isinstance(doc.page_content, dict):
            # Format the structured content
            formatted_content = "\n".join([f"{k}: {v}" for k, v in doc.page_content.items()])
            combined_results.append(formatted_content)
        else:
            combined_results.append(doc.page_content)
    
    return "\n\n".join(combined_results)

# Define the retriever tool
retriever_tool = Tool(
    name="retrieve_from_neo4j",
    description="Retrieve relevant information from the knowledge graph",
    func=retrieve_from_neo4j
)

# Define available tools
tools = [retriever_tool]

def tools_condition(state) -> Literal["tools", "end"]:
    """Determine whether to use tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, continue with tools
    if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('tool_calls'):
        return "tools"
    return "end"

# Define the graph state with proper message handling
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] 
    chat_history: List[Tuple[str, str]]
    metadata: Dict[str, Any]
    retrieval_attempts: int = 0

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """Grade document relevance and decide next action."""
    print("---CHECK RELEVANCE---")
    
    class Grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    
    # Grade using structured output
    llm_with_tool = llm.with_structured_output(Grade)
    prompt = ChatPromptTemplate.from_template("""
        You are grading document relevance to a question.
        Document: {context}
        Question: {question}
        Grade as 'yes' if relevant, 'no' if not.
    """)
    
    chain = prompt | llm_with_tool
    result = chain.invoke({"question": question, "context": docs})
    
    return "generate" if result.binary_score == "yes" else "rewrite"

def agent(state):
    """Core agent that processes messages and decides actions."""
    print("---AGENT PROCESSING---")
    messages = state["messages"]
    model = llm.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

def rewrite_query(state):
    """Transform query for better retrieval."""
    print("---QUERY TRANSFORMATION---")
    messages = state["messages"]
    question = messages[0].content
    chat_history = state.get("chat_history", [])  # Get chat history with empty default
    
    prompt = ChatPromptTemplate.from_template("""
        Improve this question for better retrieval while:
        1. Maintaining original semantic meaning
        2. Expanding ambiguous terms
        3. Considering chat history context: {chat_history}
        
        Question: {question}
        Improved version:
    """)
    
    chain = prompt | llm
    response = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
    return {"messages": [response]}

def generate_response(state):
    """Generate final response from relevant documents."""
    print("---RESPONSE GENERATION---")
    messages = state["messages"]
    question = messages[0].content
    context = messages[-1].content
    
    prompt = ChatPromptTemplate.from_template("""
        Answer based on this context:
        Question: {question}
        Context: {context}
        Answer:
    """)
    
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    return {"messages": [response]}

# Fix the tools_condition function to use END constant
def tools_condition(state) -> Literal["tools", "end"]:
    """Determine whether to use tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, continue with tools
    if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('tool_calls'):
        return "tools"
    return END 

def create_agentic_rag():
    """Create and configure the agentic RAG workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes with verbose logging
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", ToolNode(
        tools=[retriever_tool]
    ))
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate_response)
    
    # Configure edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END  # Use END constant consistently
        }
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")
    
    app = workflow.compile(checkpointer=MemorySaver())
    app.step_timeout = 30  # Increase timeout to avoid potential race conditions
    return app

def query_agentic_rag(question: str):
    """Execute agentic RAG query."""
    graph = create_agentic_rag()
    
    print("Creating initial state...")
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "chat_history": [],
        "metadata": {"debug": True},
        "retrieval_attempts": 0
    }
    
    print("Invoking graph...")
    try:
        # Use RunnableConfig for proper configuration
        from langchain_core.runnables import RunnableConfig
        
        config = RunnableConfig(
            recursion_limit=25,  # Set recursion limit if needed
            configurable={
                "thread_id": "rag_session",
                "metadata": {"debug": True}
            }
        )
        
        result = graph.invoke(initial_state, config)
        print("Graph execution completed.")
        return result["messages"][-1].content
    except Exception as e:
        print(f"Error during graph execution: {str(e)}")
        raise

# Example usage
question = "What happens in the first scene of Faust?"
response = query_agentic_rag(question)
print(response)


