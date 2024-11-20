from typing import List, Dict, Any, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict
import networkx as nx
import json
import logging
from config import get_llm, RATE_LIMIT_CONFIG, LLM_PROVIDER, get_text_processing_config
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextProcessorState(TypedDict):
    """Schema for the text processor state"""
    text: str
    format_info: str | None
    structure: Dict[str, Any] | None
    normalized_text: str | None
    chunks: List[str] | None
    normalized_chunks: List[str] | None

class TextProcessor:
    def __init__(self, llm):
        self.llm = llm
        # Get configuration
        self.config = get_text_processing_config()
        # Update the prompts to properly escape template variables
        self.format_prompt = ChatPromptTemplate.from_template("""
            Analyze the following text and identify its format:
            {text}
            """)
            
        self.structure_prompt = ChatPromptTemplate.from_template("""
            Extract the structure from this text:
            {text}
            
            Include any {{acts}}, scenes, characters, and dialogues.
            """)
            
        self.normalize_prompt = ChatPromptTemplate.from_template("""
            Normalize this text while preserving its meaning:
            {text}
            
            Include any {{characters}} and their dialogues.
            """)
        
        self.graph = StateGraph(state_schema=TextProcessorState)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Define preprocessing nodes
        self.graph.add_node("split_text", self.split_text)
        self.graph.add_node("detect_format", self.detect_format)
        self.graph.add_node("extract_structure", self.extract_structure)
        self.graph.add_node("normalize_text", self.normalize_text)
        self.graph.add_node("merge_chunks", self.merge_chunks)
        
        # Define edges
        self.graph.add_edge(START, "split_text")
        self.graph.add_edge("split_text", "detect_format")
        self.graph.add_edge("detect_format", "extract_structure")
        self.graph.add_edge("extract_structure", "normalize_text")
        self.graph.add_edge("normalize_text", "merge_chunks")
        self.graph.add_edge("merge_chunks", END)

    def split_text(self, state: TextProcessorState) -> TextProcessorState:
        """Split text into manageable chunks"""
        logger.info("Splitting text into chunks...")
        chunks = self.text_splitter.split_text(state["text"])
        
        # Apply chunk limit if configured
        if self.config["enabled"] and self.config["max_chunks"]:
            chunks = chunks[:self.config["max_chunks"]]
            logger.info(f"Testing mode: limited to {len(chunks)} chunks")
            
        state["chunks"] = chunks
        logger.info(f"Split text into {len(chunks)} chunks")
        return state

    def detect_format(self, state: TextProcessorState) -> TextProcessorState:
        """Detect format for each chunk"""
        logger.info("Detecting format for chunks...")
        format_info = []
        
        for i, chunk in enumerate(state["chunks"]):
            logger.info(f"Processing chunk {i+1}/{len(state['chunks'])}")
            prompt = self.format_prompt
            
            chain = prompt | self.llm
            result = chain.invoke({"text": chunk})
            format_info.append(result.content)
        
        # Combine format information
        state["format_info"] = "\n".join(format_info)
        return state

    def extract_structure(self, state: TextProcessorState) -> TextProcessorState:
        """Extract structure from each chunk"""
        logger.info("Extracting structure from chunks...")
        structures = []
        
        # Fixed prompt with properly escaped JSON template
        schema_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the following elements from the theatrical text into a strict JSON format:
                    {{
                      "acts": [{{"number": "int", "title": "string"}}],
                      "scenes": [{{"act": "int", "number": "int", "location": "string"}}],
                      "characters": [{{"name": "string", "first_appearance": "string"}}],
                      "relationships": [{{"source": "string", "target": "string", "type": "string"}}],
                      "locations": [{{"name": "string", "first_appearance": "string"}}],
                      "props": [{{"name": "string", "used_by": "string", "scene": "string"}}],
                      "dialogues": [{{"speaker": "string", "text": "string", "scene": "string"}}]
                    }}"""),
            ("human", "Please analyze this text and extract the structured information: {text}")
        ])
        
        # Reduce chunk size for processing
        max_chunk_size = 1000  # Reduced from 2000
        
        for i, chunk in enumerate(state["chunks"]):
            logger.info(f"Processing chunk {i+1}/{len(state['chunks'])}")
            
            # Split large chunks if needed
            if len(chunk) > max_chunk_size:
                sub_chunks = [chunk[i:i+max_chunk_size] for i in range(0, len(chunk), max_chunk_size)]
                logger.info(f"Split chunk {i+1} into {len(sub_chunks)} sub-chunks")
            else:
                sub_chunks = [chunk]
            
            chunk_structure = {
                "acts": [],
                "scenes": [],
                "characters": [],
                "relationships": [],
                "locations": [],
                "props": [],
                "dialogues": []
            }
            
            for sub_chunk in sub_chunks:
                try:
                    chain = schema_prompt | self.llm
                    result = chain.invoke({"text": sub_chunk})
                    
                    try:
                        # Extract JSON from the response
                        content = result.content
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0]
                        elif "```" in content:
                            content = content.split("```")[1].split("```")[0]
                        
                        sub_structure = json.loads(content.strip())
                        
                        # Merge sub-chunk structure into chunk structure
                        for key in chunk_structure:
                            if key in sub_structure:
                                chunk_structure[key].extend(sub_structure[key])
                                
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON for sub-chunk: {str(e)}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing sub-chunk: {str(e)}")
                    continue
            
            structures.append(chunk_structure)
        
        # Merge all structures
        try:
            state["structure"] = self.merge_structures(structures)
        except Exception as e:
            logger.error(f"Error merging structures: {str(e)}")
            state["structure"] = {
                "acts": [],
                "scenes": [],
                "characters": [],
                "relationships": [],
                "locations": [],
                "props": [],
                "dialogues": []
            }
        
        return state

    def normalize_text(self, state: TextProcessorState) -> TextProcessorState:
        """Normalize each chunk"""
        logger.info("Normalizing text chunks...")
        normalized_chunks = []
        
        # Fixed prompt without problematic template variables
        normalize_prompt = ChatPromptTemplate.from_messages([
            ("system", """Normalize this text while preserving its meaning. 
                         Keep all character names and dialogues intact."""),
            ("human", "{text}")
        ])
        
        for i, chunk in enumerate(state["chunks"]):
            logger.info(f"Processing chunk {i+1}/{len(state['chunks'])}")
            
            try:
                chain = normalize_prompt | self.llm
                result = chain.invoke({"text": chunk})
                normalized_chunks.append(result.content)
                logger.info(f"Successfully normalized chunk {i+1}")
            except Exception as e:
                logger.error(f"Error normalizing chunk {i+1}: {str(e)}")
                normalized_chunks.append(chunk)  # Use original chunk if normalization fails
        
        state["normalized_chunks"] = normalized_chunks
        return state

    def merge_chunks(self, state: TextProcessorState) -> TextProcessorState:
        """Merge normalized chunks back together"""
        logger.info("Merging normalized chunks...")
        try:
            if "normalized_chunks" not in state or not state["normalized_chunks"]:
                logger.warning("No normalized chunks found, using original chunks")
                state["normalized_text"] = "\n".join(state.get("chunks", []))
            else:
                state["normalized_text"] = "\n".join(state["normalized_chunks"])
            
            logger.info("Successfully merged chunks")
            return state
            
        except Exception as e:
            logger.error(f"Error merging chunks: {str(e)}")
            # Fallback to original text if merge fails
            state["normalized_text"] = state["text"]
            return state

    def merge_structures(self, structures: List[Dict]) -> Dict:
        """Merge multiple structure dictionaries, removing duplicates"""
        merged = {
            "acts": [],
            "scenes": [],
            "characters": [],
            "relationships": [],
            "locations": [],
            "props": [],
            "dialogues": []
        }
        
        # Use sets for deduplication
        seen_items = {
            "acts": set(),
            "characters": set(),
            "locations": set(),
            "props": set()
        }
        
        for structure in structures:
            for key in merged.keys():
                if key not in structure:
                    continue
                
                for item in structure[key]:
                    if key in seen_items:
                        # For items that need deduplication
                        item_hash = json.dumps(item, sort_keys=True)
                        if item_hash not in seen_items[key]:
                            seen_items[key].add(item_hash)
                            merged[key].append(item)
                    else:
                        # For items that can have duplicates (like dialogues)
                        merged[key].append(item)
        
        return merged

    def process(self, document: Document) -> Dict[str, Any]:
        """Main processing pipeline"""
        workflow = self.graph.compile()
        
        initial_state: TextProcessorState = {
            "text": document.page_content,
            "format_info": None,
            "structure": None,
            "normalized_text": None,
            "chunks": None,
            "normalized_chunks": None
        }
        
        try:
            logger.info("Starting text processing pipeline...")
            result = workflow.invoke(initial_state)
            
            if not result.get("normalized_text"):
                raise ValueError("Failed to normalize text")
                
            logger.info("Text processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            # Return original text if processing fails
            return {
                "text": document.page_content,
                "format_info": None,
                "structure": None,
                "normalized_text": document.page_content,
                "chunks": None,
                "normalized_chunks": None
            }

    def _fallback_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback method for text extraction when main method fails"""
        try:
            # Fixed prompt without problematic template variables
            fallback_prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract basic elements from the text in this JSON format:
                    {
                        "characters": [{"name": "string"}],
                        "locations": [{"name": "string"}],
                        "events": [{"description": "string", "participants": ["string"]}]
                    }"""),
                ("human", "{text}")
            ])
            
            chain = fallback_prompt | self.llm
            result = chain.invoke({"text": text})
            
            try:
                # Clean and parse JSON response
                content = result.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                extracted = json.loads(content.strip())
                logger.info("Fallback extraction succeeded")
                
                # Convert to standard format
                return {
                    "characters": extracted.get("characters", []),
                    "locations": extracted.get("locations", []),
                    "relationships": [],
                    "dialogues": [],
                    "props": [],
                    "acts": [],
                    "scenes": []
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse fallback extraction result: {str(e)}")
                return self._empty_structure()
                
        except Exception as e:
            logger.error(f"Error in fallback extraction: {str(e)}")
            return self._empty_structure()

    def _empty_structure(self) -> Dict[str, Any]:
        """Return empty structure with all required fields"""
        return {
            "characters": [],
            "locations": [],
            "relationships": [],
            "dialogues": [],
            "props": [],
            "acts": [],
            "scenes": []
        }