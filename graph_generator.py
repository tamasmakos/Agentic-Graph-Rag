from typing import Dict, List, Optional, Any
import networkx as nx
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from dataclasses import dataclass
from pyvis.network import Network
from langchain_anthropic import ChatAnthropic
from config import get_llm
import logging
from jinja2 import Template
import pandas as pd
from networkx.algorithms import centrality
import json
from kor import create_extraction_chain, Object, Text
from langchain.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    id: str
    type: str
    properties: Dict[str, Any]

@dataclass
class GraphRelation:
    source: str
    target: str
    type: str
    properties: Dict[str, Any]

class TheatricalGraphGenerator:
    """Generates knowledge graphs from theatrical texts"""
    
    def __init__(self, llm):
        self.llm = llm
        self.graph = nx.MultiDiGraph()
        
        # Define the dramatic structure schema using KOR
        self.dramatic_structure_schema = Object(
            id="scene_classification",
            description="Classify a scene from literature according to its dramatic structure",
            attributes=[
                Text(
                    id="dramatic_element",
                    description="The dramatic element that best describes this scene",
                    options=["Exposition", "Rising Action", "Climax", "Falling Action", "Resolution"]
                ),
                Text(
                    id="scene_description",
                    description="A brief description of what is happening in the scene"
                ),
                Text(
                    id="scene_type",
                    description="The type of scene (e.g., dialogue, monologue, action)"
                )
            ]
        )
        
        # Create the extraction chain
        self.classification_chain = create_extraction_chain(llm, self.dramatic_structure_schema)
        
        # Initialize LLMGraphTransformer with Claude
        self.llm_transformer = LLMGraphTransformer(
            llm=get_llm(),  # Use Claude instead of Groq
            strict_mode=False,
            node_properties=True,
            relationship_properties=True,
            allowed_nodes=[
                "Character", "Scene", "Act", "Location", "Event", 
                "Prop", "Dialogue", "StageDirection", "Entity",
                "Setting", "Theme", "Symbol", "Motif"
            ],
            allowed_relationships=[
                ("Character", "APPEARS_IN", "Scene"),
                ("Scene", "PART_OF", "Act"),
                ("Character", "INTERACTS_WITH", "Character"),
                ("Character", "LOCATED_IN", "Location"),
                ("Character", "USES", "Prop"),
                ("Character", "SPEAKS", "Dialogue"),
                ("Scene", "CONTAINS", "StageDirection"),
                ("Character", "SPEAKS_TO", "Character"),
                ("Character", "INVOKED_BY", "Entity"),
                ("Entity", "INVOKED_BY", "Character"),
                ("Scene", "LOCATED_IN", "Location"),
                ("Location", "LOCATED_IN", "Scene"),
                ("Character", "SYMBOLIZES", "Theme"),
                ("Prop", "REPRESENTS", "Symbol"),
                ("Theme", "EXPRESSED_IN", "Scene"),
                ("Motif", "APPEARS_IN", "Scene")
            ]
        )

    def generate_graph(self, processed_text: Dict[str, Any]) -> nx.MultiDiGraph:
        """Generate graph from processed text"""
        try:
            # Create a new graph
            graph = nx.MultiDiGraph()
            
            # Extract structure
            structure = processed_text.get("structure", {})
            
            # Add scenes with content
            for scene in structure.get("scenes", []):
                if isinstance(scene, dict):
                    scene_id = f"Scene_{scene.get('act', 0)}_{scene.get('number', 0)}"
                    # Include the actual text content
                    scene['text_content'] = scene.get('description', '')
                    graph.add_node(
                        scene_id,
                        type="Scene",
                        properties=scene
                    )
            
            # Add dialogues with content
            for dialogue in structure.get("dialogues", []):
                if isinstance(dialogue, dict):
                    dialogue_id = f"Dialogue_{hash(dialogue.get('text', ''))}"
                    graph.add_node(
                        dialogue_id,
                        type="Dialogue",
                        properties=dialogue
                    )
                    
                    # Connect dialogue to speaker
                    speaker = dialogue.get("speaker")
                    if speaker:
                        if speaker not in graph:
                            graph.add_node(speaker, type="Character", properties={})
                        graph.add_edge(speaker, dialogue_id, type="SPEAKS")
            
            # Add nodes with proper typing
            for character in structure.get("characters", []):
                if isinstance(character, dict) and "name" in character:
                    graph.add_node(
                        character["name"],
                        type="Character",
                        properties=character
                    )
            
            for location in structure.get("locations", []):
                if isinstance(location, dict) and "name" in location:
                    graph.add_node(
                        location["name"],
                        type="Location",
                        properties=location
                    )
            
            # Add relationships with proper typing
            for rel in structure.get("relationships", []):
                if isinstance(rel, dict):
                    source = rel.get("source")
                    target = rel.get("target")
                    rel_type = rel.get("type", "RELATED_TO")
                    
                    if source and target:
                        # Add nodes if they don't exist
                        if source not in graph:
                            graph.add_node(source, type="Entity", properties={})
                        if target not in graph:
                            graph.add_node(target, type="Entity", properties={})
                        
                        # Add edge with properties
                        graph.add_edge(
                            source,
                            target,
                            type=rel_type,
                            properties={}
                        )
            
            return graph
            
        except Exception as e:
            logger.error(f"Error generating graph: {str(e)}")
            # Return minimal valid graph
            g = nx.MultiDiGraph()
            g.add_node("Error", type="Error", properties={"message": str(e)})
            return g

    def _fallback_extraction(self, text: str):
        """Fallback method to extract basic entities and relationships"""
        try:
            # Create a simpler prompt for basic extraction
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at analyzing literary texts.
                Extract key elements from the text in this exact JSON format:
                {
                    "characters": [{"name": "string"}],
                    "locations": [{"name": "string"}],
                    "events": [{"description": "string", "participants": ["string"]}]
                }"""),
                ("human", "{text}")
            ])
            
            chain = prompt | self.llm
            result = chain.invoke({"text": text})
            
            try:
                extracted = json.loads(result.content)
                logger.info("Fallback extraction succeeded")
                
                # Add characters
                for char in extracted.get("characters", []):
                    self.graph.add_node(char["name"], type="Character", properties={})
                    
                # Add locations
                for loc in extracted.get("locations", []):
                    self.graph.add_node(loc["name"], type="Location", properties={})
                    
                # Add events and relationships
                for event in extracted.get("events", []):
                    event_id = f"Event_{hash(event['description'])}"
                    self.graph.add_node(event_id, type="Event", 
                                      properties={"description": event["description"]})
                    
                    # Connect participants
                    for participant in event.get("participants", []):
                        if participant in self.graph:
                            self.graph.add_edge(participant, event_id, 
                                              type="PARTICIPATES_IN", properties={})
                            
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse fallback extraction result: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in fallback extraction: {str(e)}")

    def _split_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _extract_dramatic_structure(self, text: str) -> Dict[str, Any]:
        """Extract dramatic structure using KOR"""
        try:
            result = self.classification_chain.invoke({"text": text})
            return result.get("data", {}).get("scene_classification", {})
        except Exception as e:
            logger.error(f"Error extracting dramatic structure: {str(e)}")
            return {}

    def _add_to_graph(self, graph_documents: List[Any]) -> None:
        """Add nodes and relationships to the graph"""
        for doc in graph_documents:
            # Add nodes
            for node in doc.nodes:
                self.graph.add_node(
                    node.id,
                    type=node.type,
                    properties=node.properties
                )
            
            # Add relationships
            for rel in doc.relationships:
                self.graph.add_edge(
                    rel.source.id,
                    rel.target.id,
                    type=rel.type,
                    properties=rel.properties
                )

    def visualize_graph(self, highlight_nodes: Optional[List[str]] = None) -> Network:
        """Creates an interactive visualization of the graph"""
        try:
            net = Network(
                height="750px", 
                width="100%",
                notebook=False,
                directed=True,
                bgcolor="#ffffff",
                font_color="#000000"
            )
            
            # Add nodes with improved attributes
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                net.add_node(
                    node,
                    label=f"{node}\n({node_data.get('type', 'Unknown')})",
                    title=str(node_data.get('properties', {})),
                    color=self._get_node_color(node_data.get('type', 'Unknown')),
                    size=25
                )
            
            # Add edges with improved attributes
            for source, target, data in self.graph.edges(data=True):
                net.add_edge(
                    source, 
                    target,
                    label=data.get('type', ''),
                    title=str(data.get('properties', {})),
                    arrows="to"
                )

            # Configure physics for better visualization
            net.set_options("""
            {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "solver": "forceAtlas2Based",
                    "stabilization": {"iterations": 100}
                }
            }
            """)

            return net
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return Network(height="750px", width="100%")

    def _get_node_color(self, node_type: str) -> str:
        """Return color based on node type"""
        colors = {
            "Character": "#ff7f0e",
            "Scene": "#1f77b4",
            "Act": "#2ca02c",
            "Location": "#d62728",
            "Event": "#9467bd",
            "Prop": "#8c564b",
            "Dialogue": "#e377c2",
            "StageDirection": "#7f7f7f",
            "Entity": "#bcbd22",
            "Theme": "#17becf",
            "Symbol": "#e377c2",
            "Motif": "#7f7f7f"
        }
        return colors.get(node_type, "#aec7e8")

    def generate_graph_statistics(self) -> Dict[str, pd.DataFrame]:
        """Generate and save various graph statistics"""
        logger.info("Generating graph statistics...")
        
        # Convert MultiDiGraph to DiGraph for statistics
        simple_graph = nx.DiGraph(self.graph)
        
        stats = {}
        try:
            # Node statistics
            logger.info("Calculating node statistics...")
            node_stats = {
                'degree_centrality': nx.degree_centrality(simple_graph),
                'betweenness_centrality': nx.betweenness_centrality(simple_graph),
                'pagerank': nx.pagerank(simple_graph),
            }
            
            node_df = pd.DataFrame.from_dict(node_stats)
            node_df['node_type'] = [self.graph.nodes[node].get('type', 'Unknown') 
                                   for node in node_df.index]
            stats['node_statistics'] = node_df
            
            # Add relationship statistics
            edge_types = [data['type'] for _, _, data in self.graph.edges(data=True)]
            relationship_counts = pd.Series(edge_types).value_counts()
            stats['relationship_distribution'] = pd.DataFrame(relationship_counts, 
                                                            columns=['count'])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistics: {str(e)}")
            return {}

    def _add_structure_nodes(self, structure: Dict[str, Any]) -> None:
        """Add nodes from the extracted structure"""
        try:
            # Add characters
            for character in structure.get("characters", []):
                if isinstance(character, dict):
                    name = character.get("name")
                    if name:
                        self.graph.add_node(name, type="Character", properties=character)
            
            # Add locations
            for location in structure.get("locations", []):
                if isinstance(location, dict):
                    name = location.get("name")
                    if name:
                        self.graph.add_node(name, type="Location", properties=location)
            
            # Add scenes and connect to acts
            for scene in structure.get("scenes", []):
                if isinstance(scene, dict):
                    scene_id = f"Scene_{scene.get('act', 0)}_{scene.get('number', 0)}"
                    self.graph.add_node(scene_id, type="Scene", properties=scene)
                    
                    # Connect to act
                    act_id = f"Act_{scene.get('act', 0)}"
                    self.graph.add_node(act_id, type="Act", properties={"number": scene.get("act", 0)})
                    self.graph.add_edge(scene_id, act_id, type="PART_OF")
            
            # Add relationships
            for rel in structure.get("relationships", []):
                if isinstance(rel, dict):
                    source = rel.get("source")
                    target = rel.get("target")
                    rel_type = rel.get("type")
                    if source and target and rel_type:
                        if source not in self.graph:
                            self.graph.add_node(source, type="Character", properties={})
                        if target not in self.graph:
                            self.graph.add_node(target, type="Character", properties={})
                        self.graph.add_edge(source, target, type=rel_type)
                        
        except Exception as e:
            logger.error(f"Error adding structure nodes: {str(e)}")

    def _validate_and_clean_graph(self) -> None:
        """Validate and clean the graph"""
        try:
            # Remove isolated nodes
            isolated_nodes = list(nx.isolates(self.graph))
            self.graph.remove_nodes_from(isolated_nodes)
            
            # Fix edge iteration
            for u, v, key in list(self.graph.edges(keys=True)):  # Changed this line
                if 'type' not in self.graph.edges[u, v, key]:
                    self.graph.edges[u, v, key]['type'] = 'RELATED_TO'
                if 'properties' not in self.graph.edges[u, v, key]:
                    self.graph.edges[u, v, key]['properties'] = {}
                    
            # Ensure all nodes have required attributes
            for node in self.graph.nodes():
                if 'type' not in self.graph.nodes[node]:
                    self.graph.nodes[node]['type'] = 'Unknown'
                if 'properties' not in self.graph.nodes[node]:
                    self.graph.nodes[node]['properties'] = {}
                    
        except Exception as e:
            logger.error(f"Error validating graph: {str(e)}")
            # Ensure minimum valid structure even after error
            if len(self.graph.nodes()) == 0:
                self.graph.add_node(
                    "Document",
                    type="Document",
                    properties={"description": "Fallback document node"}
                )

    def _create_minimal_graph(self, processed_text: Dict[str, Any]) -> None:
        """Create a minimal valid graph from processed text"""
        try:
            # Add a default node
            self.graph.add_node(
                "Text",
                type="Document",
                properties={"content": processed_text.get("normalized_text", "Empty document")}
            )
            
            # Add structure if available
            structure = processed_text.get("structure", {})
            if structure:
                self._add_structure_nodes(structure)
                
        except Exception as e:
            logger.error(f"Error creating minimal graph: {str(e)}")

    def _process_chunk(self, chunk: str) -> Dict:
        """Process a chunk with length limits"""
        MAX_CHUNK_LENGTH = 3000  # Reduced from previous value
        
        if len(chunk) > MAX_CHUNK_LENGTH:
            # Split into smaller chunks
            chunks = [chunk[i:i + MAX_CHUNK_LENGTH] 
                     for i in range(0, len(chunk), MAX_CHUNK_LENGTH)]
            results = []
            for sub_chunk in chunks:
                try:
                    result = self._process_single_chunk(sub_chunk)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing sub-chunk: {str(e)}")
            return self._merge_chunk_results(results)
        else:
            return self._process_single_chunk(chunk)