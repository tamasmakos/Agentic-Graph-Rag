from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a wrapper class for embeddings
class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        # Extract the 'question' from the input if it's a dictionary
        if isinstance(text, dict) and 'question' in text:
            text = text['question']
        
        # Ensure text is a string
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text)}")
        
        # Convert to list for encoding
        text_list = [text]
        
        # Encode the text and return the embedding
        embeddings = self.model.encode(text_list)
        
        # Return the embedding as a list
        return embeddings[0].tolist()

# Initialize embeddings model
embeddings = LocalHuggingFaceEmbeddings('all-MiniLM-L6-v2')