from abc import ABC, abstractmethod
from typing import Dict, Any, List
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = setup_logger(f"Agent.{name}")
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return result"""
        pass
    
    def log_action(self, action: str, details: str = ""):
        """Log agent action"""
        self.logger.info(f"{self.name} - {action}: {details}") 