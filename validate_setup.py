import os
from dotenv import load_dotenv
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import GROQ_API_KEY, LLM_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_setup():
    """Validate the environment setup and API key"""
    try:
        logger.info(f"API Key found: {'Yes' if GROQ_API_KEY else 'No'}")
        logger.info(f"API Key length: {len(GROQ_API_KEY)}")
        logger.info(f"API Key prefix: {GROQ_API_KEY[:4] + '...'}")
        
        # Test LLM connection
        logger.info("Testing LLM connection...")
        llm = ChatGroq(**LLM_CONFIG)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Test connection"),
            ("human", "Hello")
        ])
        
        logger.info("Sending test request to Groq API...")
        chain = prompt | llm
        response = chain.invoke({"text": "Hello"})
        
        logger.info("API connection successful")
        logger.info("All validations passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        if "Invalid API Key" in str(e):
            logger.error("Please check that your API key is correctly formatted and valid")
            logger.error("You can get a new API key from: https://console.groq.com/keys")
        return False

if __name__ == "__main__":
    validate_setup() 