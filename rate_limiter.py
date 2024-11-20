"""Custom rate limiter implementation"""
import time
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseRateLimiter(ABC):
    """Abstract base class for rate limiters"""
    
    @abstractmethod
    def acquire(self) -> None:
        """Acquire permission to make a request"""
        pass

class CustomRateLimiter(BaseRateLimiter):
    """Custom rate limiter with exponential backoff"""
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.last_request_time = 0
        self.min_interval = 60.0 / requests_per_minute
        self.retry_count = 0

    def acquire(self) -> None:
        """Acquire permission to make a request with exponential backoff"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            if self.retry_count > 0:
                sleep_time *= (2 ** self.retry_count)  # Exponential backoff
            logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            self.retry_count += 1
        else:
            self.retry_count = 0
            
        self.last_request_time = time.time() 