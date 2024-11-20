"""
Knowledge Graph Generator for Theatrical Texts
"""
import os
from typing import Dict, List, Optional, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from text_processor import TextProcessor
from graph_generator import TheatricalGraphGenerator
from rate_limiter import CustomRateLimiter
import logging
from langchain.prompts import ChatPromptTemplate
import time
from config import get_llm, LLM_CONFIG
from graph_rag import GraphRetriever
from pyvis.network import Network
import networkx as nx
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaustKGGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Faust Knowledge Graph Generator"""
        logger.info("Initializing FaustKGGenerator...")
        
        # Use batch processing for rate limiting
        self.rate_limiter = CustomRateLimiter(requests_per_minute=60)  # Increased limit
        
        # Initialize LLM with caching
        self.llm = get_llm()
        
        # Initialize components with shared LLM instance
        self.text_processor = TextProcessor(self.llm)
        self.graph_generator = TheatricalGraphGenerator(self.llm)
        
        # Pre-initialize graph retriever
        self.graph_retriever = None
        
        # Cache for processed chunks
        self.chunk_cache = {}

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text and generate knowledge graph"""
        try:
            # Process in parallel chunks
            from concurrent.futures import ThreadPoolExecutor
            import concurrent.futures
            
            # Chunk the text more efficiently
            chunks = self._smart_chunk_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Process chunks in parallel
            logger.info("Starting parallel chunk processing...")
            processed_chunks = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_chunk = {
                    executor.submit(self._process_chunk, chunk): i 
                    for i, chunk in enumerate(chunks)
                }
                
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        chunk_result = future.result()
                        if chunk_result:
                            processed_chunks.append(chunk_result)
                            logger.info(f"Successfully processed chunk {chunk_idx + 1}")
                        else:
                            logger.warning(f"Chunk {chunk_idx + 1} processing returned None")
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")

            # Merge results
            logger.info("Merging processed chunks...")
            merged_result = self._merge_chunk_results(processed_chunks)
            logger.info("Successfully merged chunks")
            
            # Generate graph
            logger.info("Generating graph from merged results...")
            graph = self.graph_generator.generate_graph(merged_result)
            logger.info(f"Generated graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
            
            # Create visualization
            logger.info("Creating graph visualization...")
            visualization = self._create_visualization(graph)
            logger.info("Successfully created visualization")
            
            # Initialize retriever with proper error handling
            try:
                if graph and self.llm:
                    self.graph_retriever = GraphRetriever(graph=graph, llm=self.llm)
                    logger.info("Successfully initialized graph retriever")
                else:
                    logger.warning("Missing graph or LLM, skipping retriever initialization")
                    self.graph_retriever = None
            except Exception as e:
                logger.error(f"Failed to initialize graph retriever: {str(e)}")
                self.graph_retriever = None
            
            return {
                "processed_text": merged_result,
                "graph": graph,
                "visualization": visualization,
                "statistics": self._generate_stats(graph)
            }
            
        except Exception as e:
            logger.error(f"Error in process_text: {str(e)}")
            raise

    def _smart_chunk_text(self, text: str) -> List[str]:
        """Chunk text more intelligently based on natural boundaries"""
        import re
        
        # Try to split on scene/chapter boundaries first
        scene_chunks = re.split(r'\n(?=SCENE|ACT|Chapter)', text)
        
        if len(scene_chunks) > 1:
            return scene_chunks
            
        # Fallback to paragraph chunks
        para_chunks = re.split(r'\n\n+', text)
        
        if len(para_chunks) > 1:
            return para_chunks
            
        # Final fallback to fixed size chunks
        return [text[i:i+2000] for i in range(0, len(text), 2000)]

    def _process_chunk(self, chunk: str) -> Optional[Dict]:
        """Process a single chunk with caching"""
        chunk_hash = hash(chunk)
        
        if chunk_hash in self.chunk_cache:
            return self.chunk_cache[chunk_hash]
            
        try:
            processed = self.text_processor.process(Document(page_content=chunk))
            self.chunk_cache[chunk_hash] = processed
            return processed
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            return None

    def _merge_chunk_results(self, results: List[Dict]) -> Dict:
        """Merge processed chunk results efficiently"""
        merged = {
            "normalized_text": "",
            "structure": {
                "acts": [],
                "scenes": [],
                "characters": set(),  # Use sets for deduplication
                "relationships": [],  # Keep as list since relationships can be duplicate
                "locations": set(),
                "props": set(),
                "dialogues": []  # Keep as list to maintain order
            }
        }
        
        for result in results:
            if not result:
                logger.warning("Skipping None result in merge_chunk_results")
                continue
                
            # Add normalized text if available
            merged["normalized_text"] += result.get("normalized_text", "") + "\n"
            
            # Safely get structure, defaulting to empty dict if None
            structure = result.get("structure") or {}
            
            # Safely extend lists and update sets
            try:
                # Lists - append all
                merged["structure"]["acts"].extend(structure.get("acts", []))
                merged["structure"]["scenes"].extend(structure.get("scenes", []))
                merged["structure"]["relationships"].extend(structure.get("relationships", []))
                merged["structure"]["dialogues"].extend(structure.get("dialogues", []))
                
                # Sets - convert to strings for hashability
                for character in structure.get("characters", []):
                    if isinstance(character, dict):
                        merged["structure"]["characters"].add(json.dumps(character, sort_keys=True))
                
                for location in structure.get("locations", []):
                    if isinstance(location, dict):
                        merged["structure"]["locations"].add(json.dumps(location, sort_keys=True))
                    
                for prop in structure.get("props", []):
                    if isinstance(prop, dict):
                        merged["structure"]["props"].add(json.dumps(prop, sort_keys=True))
                    
            except Exception as e:
                logger.error(f"Error merging structure: {str(e)}")
                continue
        
        # Convert sets back to lists and parse JSON strings back to dicts
        try:
            merged["structure"]["characters"] = [json.loads(c) for c in merged["structure"]["characters"]]
            merged["structure"]["locations"] = [json.loads(l) for l in merged["structure"]["locations"]]
            merged["structure"]["props"] = [json.loads(p) for p in merged["structure"]["props"]]
        except Exception as e:
            logger.error(f"Error converting sets to lists: {str(e)}")
        
        return merged

    def _create_visualization(self, graph: nx.MultiDiGraph, highlighted_nodes: List[str] = None, highlighted_edges: List[tuple] = None) -> Network:
        """Create an interactive visualization of the graph"""
        logger.info("Creating graph visualization...")
        
        try:
            # Create pyvis network with specific settings
            net = Network(
                height="750px",
                width="100%",
                bgcolor="#222222",
                font_color="white",
                directed=True,
                notebook=False
            )

            # Add nodes
            for node in graph.nodes():
                node_data = graph.nodes[node]
                node_type = node_data.get('type', 'Unknown')
                is_highlighted = highlighted_nodes and node in highlighted_nodes
                
                net.add_node(
                    node,
                    label=f"{node}\n({node_type})",
                    title=str(node_data.get('properties', {})),
                    color="#ff3333" if is_highlighted else self._get_node_color(node_type),
                    size=30 if is_highlighted else 20,
                    borderWidth=3 if is_highlighted else 1
                )

            # Add edges
            for source, target, data in graph.edges(data=True):
                is_highlighted = highlighted_edges and (source, target) in highlighted_edges
                
                net.add_edge(
                    source,
                    target,
                    label=data.get('type', 'RELATED_TO'),
                    title=str(data.get('properties', {})),
                    color="#ff3333" if is_highlighted else "#ffffff",
                    width=3 if is_highlighted else 1
                )

            # Configure physics
            net.set_options("""
            {
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -2000,
                        "centralGravity": 0.3,
                        "springLength": 200
                    },
                    "minVelocity": 0.75
                }
            }
            """)

            # Save to temporary file and read back
            temp_path = "temp_viz.html"
            net.save_graph(temp_path)
            with open(temp_path, 'r', encoding='utf-8') as f:
                net.html = f.read()
            os.remove(temp_path)
            
            return net
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            # Return minimal working network
            fallback_net = Network(height="750px", width="100%", notebook=False)
            fallback_net.add_node(1, label="Error", title=f"Visualization failed: {str(e)}")
            return fallback_net

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

    def query_graph(self, query: str, graph: nx.MultiDiGraph) -> str:
        """Query the knowledge graph"""
        if not self.graph_retriever:
            self.graph_retriever = GraphRetriever(graph, self.llm)
            
        try:
            # Log the query
            logger.info(f"\n{'='*50}\nProcessing Query: {query}\n{'='*50}")
            
            # Get relevant documents with detailed logging
            docs = self.graph_retriever.get_relevant_documents(query)
            logger.info(f"\nFound {len(docs)} relevant documents")
            
            # Log raw document data
            logger.info("\nRaw Document Data:")
            for i, doc in enumerate(docs):
                logger.info(f"\nDocument {i+1}:")
                logger.info(f"Raw Content: {doc.page_content}")
                logger.info(f"Metadata: {json.dumps(doc.metadata, indent=2)}")
            
            # Create context from documents
            context = "\n".join(doc.page_content for doc in docs)
            logger.info(f"\nCombined Context for LLM:\n{context}")
            
            # Log the prompt that will be sent to LLM
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Based on the following graph information, answer the user's question.
                If you cannot find a direct answer, try to infer from the relationships.
                Always cite the specific nodes and relationships you used.
                
                Format your response like this:
                Answer: [Your detailed answer here]
                
                Used Information:
                - Nodes: [List all nodes you used]
                - Relationships: [List all relationships you used]
                - Reasoning: [Explain how you arrived at your answer]
                """),
                ("human", "Context: {context}\nQuestion: {question}")
            ])
            
            # Log the formatted prompt
            formatted_prompt = prompt.format(context=context, question=query)
            logger.info(f"\nFormatted Prompt to LLM:\n{formatted_prompt}")
            
            # Generate answer
            chain = prompt | self.llm
            logger.info("\nGenerating answer...")
            result = chain.invoke({"context": context, "question": query})
            
            # Log the complete response
            logger.info(f"\nLLM Raw Response:\n{result.content}")
            
            return result.content
            
        except Exception as e:
            logger.error(f"Error querying graph: {str(e)}")
            return f"Error querying graph: {str(e)}"

    def _generate_stats(self, graph) -> Dict[str, Any]:
        """Generate graph statistics"""
        try:
            stats = {
                "node_count": len(graph.nodes()),
                "edge_count": len(graph.edges()),
                "density": nx.density(graph),
                "node_types": {},
                "edge_types": {},
                "connected_components": nx.number_connected_components(graph.to_undirected())
            }
            
            # Count node types
            for node in graph.nodes():
                node_type = graph.nodes[node].get('type', 'Unknown')
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
                
            # Fix edge type counting for MultiDiGraph
            for u, v, key in graph.edges(keys=True):  # Changed this line
                edge_type = graph.edges[u, v, key].get('type', 'unknown')
                stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1
                
            logger.info(f"Generated graph stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error generating graph stats: {str(e)}")
            return {
                "node_count": 0,
                "edge_count": 0,
                "error": str(e)
            }

    def create_highlighted_visualization(self, graph: nx.MultiDiGraph, highlighted_nodes: List[str]) -> Network:
        """Create visualization with highlighted nodes"""
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            directed=True,
            notebook=False,
            cdn_resources="remote"
        )
        
        # Add nodes with highlighting
        for node in graph.nodes():
            is_highlighted = node in highlighted_nodes
            color = "#ff3333" if is_highlighted else self._get_node_color(graph.nodes[node].get('type', 'Unknown'))
            size = 30 if is_highlighted else 20
            
            net.add_node(
                node,
                label=str(node),
                title=f"Type: {graph.nodes[node].get('type', 'Unknown')}",
                color=color,
                size=size
            )
        
        # Add edges
        for source, target, data in graph.edges(data=True):
            net.add_edge(
                source,
                target,
                title=data.get('type', 'RELATED_TO'),
                label=data.get('type', 'RELATED_TO'),
                physics=True
            )
        
        # Configure physics
        net.set_options("""
        {
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -2000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 0.1
                },
                "maxVelocity": 50,
                "minVelocity": 0.1,
                "solver": "barnesHut"
            }
        }""")
        
        return net

def main():
    """Main entry point"""
    try:
        logger.info("Starting Faust KG Generator...")
        
        # Set testing mode
        os.environ["TESTING"] = "true"
        
        # Initialize generator
        generator = FaustKGGenerator()

        logger.info("Reading input text...")
        with open("faust.txt", "r", encoding="utf-8") as f:
            text = f.read()
            
        logger.info("Processing text and generating graph...")
        result = generator.process_text(text)  # Text truncation happens inside process_text

        # Save visualization with proper error handling
        try:
            output_file = "faust_graph.html"
            result["visualization"].show(output_file)
            logger.info(f"Visualization saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save visualization: {str(e)}")

        # Print statistics with proper formatting
        if result.get("statistics"):
            print("\n=== Graph Statistics ===")
            for key, value in result["statistics"].items():
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for subkey, subvalue in value.items():
                        print(f"  {subkey}: {subvalue}")
                else:
                    print(f"{key}: {value}")
        else:
            logger.warning("No statistics generated")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 