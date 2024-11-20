import streamlit as st
from faust_kg_gen import FaustKGGenerator
import logging
import os
from config import get_text_processing_config
import networkx as nx
import io
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize generator with cache_resource
@st.cache_resource
def get_generator():
    """Initialize and cache the FaustKGGenerator"""
    try:
        generator = FaustKGGenerator()
        logger.info("FaustKGGenerator initialized and cached")
        return generator
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

# Modified process_text function with underscore prefix for generator
@st.cache_data(show_spinner=False)
def process_text(_generator, text: str):
    """Process text without caching the generator"""
    try:
        # Get text processing config
        config = get_text_processing_config()
        if config["enabled"]:
            text = text[config["start_idx"]:config["end_idx"]]
            st.info(f"Testing mode: processing text from index {config['start_idx']} to {config['end_idx']}")
        
        result = _generator.process_text(text)
        
        # Convert visualization to HTML string
        if "visualization" in result:
            viz = result["visualization"]
            temp_path = "temp_viz.html"
            viz.save_graph(temp_path)
            with open(temp_path, 'r', encoding='utf-8') as f:
                viz_html = f.read()
            os.remove(temp_path)
            result["visualization_html"] = viz_html
            del result["visualization"]  # Remove non-serializable object
        
        # Convert graph to serializable format
        if "graph" in result:
            graph = result["graph"]
            result["graph_data"] = {
                "nodes": list(graph.nodes(data=True)),
                "edges": list(graph.edges(data=True))
            }
            del result["graph"]  # Remove non-serializable object
        
        return result
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise

def main():
    # Set page config to wide mode
    st.set_page_config(layout="wide", page_title="Literary Knowledge Graph Chat")
    
    # Create three columns with ratio 1:1:2
    col1, col2, col3 = st.columns([1, 1, 2])
    
    # Left Column - Document Management
    with col1:
        st.title("Documents")
        
        # Initialize generator once
        generator = get_generator()
        
        # File upload
        uploaded_file = st.file_uploader("Upload Text File", type=['txt'])
        
        # Display uploaded documents
        st.subheader("Uploaded Documents")
        if uploaded_file:
            st.success(f"📄 {uploaded_file.name}")
            
            # Document info
            file_size = len(uploaded_file.getvalue()) / 1024  # Size in KB
            st.info(f"Size: {file_size:.1f} KB")
            
            # Create a placeholder for real-time logs
            log_placeholder = st.empty()
            
            # Process button
            if st.button("Process Document"):
                # Initialize session state
                if 'processed_results' not in st.session_state:
                    st.session_state.processed_results = None
                if 'log_messages' not in st.session_state:
                    st.session_state.log_messages = []
                
                # Create a custom StreamlitHandler for real-time logging
                class StreamlitHandler(logging.Handler):
                    def emit(self, record):
                        log_entry = self.format(record)
                        with log_placeholder:
                            current_logs = st.session_state.get('current_logs', [])
                            current_logs.append(log_entry)
                            # Show logs in a scrollable text area
                            st.text_area("Processing Logs", 
                                       value="\n".join(current_logs),
                                       height=400,
                                       key="log_area")
                            st.session_state.current_logs = current_logs

                # Configure the custom handler
                streamlit_handler = StreamlitHandler()
                streamlit_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                )
                logger.addHandler(streamlit_handler)
                
                # Read text
                text = uploaded_file.getvalue().decode('utf-8')
                
                # Process text
                with st.spinner("Processing..."):
                    try:
                        result = process_text(generator, text)
                        st.session_state.processed_results = result
                        st.success("Processing complete!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Processing error: {str(e)}", exc_info=True)
                    finally:
                        # Remove the handler
                        logger.removeHandler(streamlit_handler)
                        # Clear the logs from session state
                        if 'current_logs' in st.session_state:
                            del st.session_state.current_logs
    
    # Middle Column - Chat Interface
    with col2:
        st.title("Chat")
        
        if 'processed_results' in st.session_state and st.session_state.processed_results:
            # Display statistics in collapsible section
            with st.expander("Graph Statistics", expanded=False):
                if st.session_state.processed_results.get("statistics"):
                    st.json(st.session_state.processed_results["statistics"])
            
            # Chat interface
            st.subheader("Ask Questions")
            user_question = st.text_input("Your question:", key="question_input")
            
            if user_question:
                with st.spinner("Searching..."):
                    try:
                        # Create log capture for query
                        query_log_stream = io.StringIO()
                        query_handler = logging.StreamHandler(query_log_stream)
                        query_handler.setFormatter(
                            logging.Formatter('%(message)s')  # Simplified format for readability
                        )
                        logger.addHandler(query_handler)
                        
                        # Reconstruct graph for querying
                        if "graph_data" in st.session_state.processed_results:
                            graph = nx.MultiDiGraph()
                            for node, data in st.session_state.processed_results["graph_data"]["nodes"]:
                                graph.add_node(node, **data)
                            for source, target, data in st.session_state.processed_results["graph_data"]["edges"]:
                                graph.add_edge(source, target, **data)
                            
                            # Get answer and logs
                            answer = generator.query_graph(user_question, graph)
                            query_logs = query_log_stream.getvalue()
                            
                            # Display answer
                            st.markdown("### Answer")
                            st.write(answer)
                            
                            # Extract mentioned nodes and relationships from the "Used Information" section
                            mentioned_nodes = set()
                            mentioned_edges = set()
                            
                            if "Used Information:" in answer:
                                info_section = answer.split("Used Information:")[1].split("Reasoning:")[0]
                                
                                # Extract nodes
                                if "Nodes:" in info_section:
                                    nodes_text = info_section.split("Nodes:")[1].split("Relationships:")[0]
                                    nodes = [n.strip() for n in nodes_text.replace("(", "").split(")")[:-1]]
                                    mentioned_nodes.update(n.split()[0] for n in nodes)
                                
                                # Extract relationships
                                if "Relationships:" in info_section:
                                    rels_text = info_section.split("Relationships:")[1].strip()
                                    for rel in rels_text.split("\n"):
                                        if "->" in rel:
                                            source = rel.split("->")[0].split("(")[0].strip()
                                            target = rel.split("->")[1].split("(")[0].strip()
                                            mentioned_nodes.add(source)
                                            mentioned_nodes.add(target)
                                            mentioned_edges.add((source, target))
                            
                            # Create highlighted visualization
                            viz = generator._create_visualization(
                                graph, 
                                highlighted_nodes=list(mentioned_nodes),
                                highlighted_edges=list(mentioned_edges)
                            )
                            
                            # Update visualization in session state
                            temp_path = "temp_viz.html"
                            viz.save_graph(temp_path)
                            with open(temp_path, 'r', encoding='utf-8') as f:
                                st.session_state.processed_results["visualization_html"] = f.read()
                            os.remove(temp_path)
                            
                            # Display query processing details
                            with st.expander("Query Processing Details", expanded=True):
                                st.markdown("#### Retrieved Information and Reasoning")
                                st.code(query_logs)
                            
                        else:
                            st.warning("No graph data available. Please process the document first.")
                            
                    except Exception as e:
                        st.error(f"Query error: {str(e)}")
                        logger.error(f"Query error: {str(e)}", exc_info=True)
                    finally:
                        # Remove query handler
                        logger.removeHandler(query_handler)
                        query_log_stream.close()
        else:
            st.info("Please upload and process a document first.")
    
    # Right Column - Visualization
    with col3:
        st.title("Knowledge Graph")
        
        if 'processed_results' in st.session_state:
            try:
                if "visualization_html" in st.session_state.processed_results:
                    st.components.v1.html(
                        st.session_state.processed_results["visualization_html"],
                        height=800,
                        scrolling=True
                    )
                else:
                    st.info("Processing the visualization...")
                    
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
                logger.error(f"Visualization error: {str(e)}", exc_info=True)
        else:
            st.info("Visualization will appear here after processing a document.")

if __name__ == "__main__":
    main()