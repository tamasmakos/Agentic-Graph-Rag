from typing import Optional, Dict, Any
import streamlit as st
from faust_kg_gen import FaustKGGenerator
import logging
import os
from config import get_llm
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components with caching
@st.cache_resource
def init_generator():
    """Initialize the FaustKGGenerator (cached)"""
    try:
        generator = FaustKGGenerator()
        logger.info("FaustKGGenerator initialized and cached")
        return generator
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise

@st.cache_data
def process_text_cached(generator, text_content: str, testing_mode: bool) -> Dict[str, Any]:
    """Process text and generate knowledge graph (cached)"""
    try:
        logger.info("Starting cached text processing...")
        os.environ["TESTING"] = str(testing_mode).lower()
        
        # If testing mode, truncate text
        if testing_mode:
            text_content = text_content[:100000]
            logger.info(f"Testing mode: truncated text to {len(text_content)} characters")
        
        # Process text
        logger.info("Calling generator.process_text...")
        result = generator.process_text(text_content)
        logger.info("Text processing completed")
        
        # Validate result
        if not isinstance(result, dict):
            logger.error(f"Invalid result type: {type(result)}")
            raise ValueError("Invalid result format")
            
        required_keys = ["graph", "visualization", "statistics"]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            logger.error(f"Missing required keys in result: {missing_keys}")
            raise ValueError(f"Missing required data: {', '.join(missing_keys)}")
        
        # Store the processed result in session state
        logger.info("Storing result in session state...")
        st.session_state['kg_result'] = result
        st.session_state['graph'] = result['graph']
        
        return result
        
    except Exception as e:
        logger.error(f"Error in process_text_cached: {str(e)}", exc_info=True)
        raise

def query_graph(generator, query: str) -> str:
    """Query the knowledge graph using cached graph"""
    try:
        if 'graph' not in st.session_state:
            raise ValueError("Please generate the knowledge graph first")
            
        return generator.query_graph(query=query, graph=st.session_state['graph'])
    except Exception as e:
        logger.error(f"Error querying graph: {str(e)}")
        return f"Error: {str(e)}"

def main():
    st.title("Literature Knowledge Graph Generator")
    
    # Initialize generator (cached)
    try:
        generator = init_generator()
        logger.info("Successfully initialized generator")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {str(e)}")
        st.error("Failed to initialize the application. Please check the logs.")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Generate Graph", "Query Graph"])
    
    with tab1:
        if uploaded_file:
            # Read file content
            text_content = uploaded_file.read().decode()
            
            # Testing mode toggle
            testing_mode = st.checkbox("Testing mode (process first 5000 characters only)", value=True)
            
            if st.button("Generate Knowledge Graph"):
                try:
                    with st.spinner("Processing text..."):
                        # Use cached processing
                        logger.info("Starting text processing...")
                        result = process_text_cached(generator, text_content, testing_mode)
                        logger.info("Text processing completed")
                    
                    # Display statistics
                    if result.get("statistics"):
                        st.subheader("Graph Statistics")
                        logger.info("Displaying statistics...")
                        st.json(result["statistics"])
                    
                    # Display visualization
                    st.subheader("Knowledge Graph Visualization")
                    logger.info("Preparing visualization...")
                    
                    # Check visualization object
                    if "visualization" not in result:
                        logger.error("No visualization object in result")
                        st.error("Failed to create visualization")
                        return
                        
                    viz = result["visualization"]
                    logger.info(f"Visualization object type: {type(viz)}")
                    
                    if not hasattr(viz, 'html'):
                        logger.error("Visualization object has no html attribute")
                        st.error("Invalid visualization format")
                        return
                        
                    logger.info(f"HTML content length: {len(viz.html) if viz.html else 0}")
                    
                    try:
                        st.components.v1.html(
                            viz.html,
                            height=800,
                            scrolling=True
                        )
                        logger.info("Successfully displayed visualization")
                    except Exception as e:
                        logger.error(f"Failed to render visualization: {str(e)}")
                        st.error("Failed to display the graph visualization")
                    
                except Exception as e:
                    logger.error(f"Processing error: {str(e)}", exc_info=True)
                    st.error(f"Error processing text: {str(e)}")
    
    with tab2:
        if 'graph' in st.session_state:
            query = st.text_input("Enter your query about the text:")
            if query:
                with st.spinner("Querying knowledge graph..."):
                    answer = query_graph(generator, query)
                    st.write("Answer:", answer)
        else:
            st.warning("Please generate a knowledge graph first")

if __name__ == "__main__":
    main() 