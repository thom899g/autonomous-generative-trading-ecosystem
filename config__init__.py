"""
Configuration module for the Autonomous Generative Trading Ecosystem.
Centralizes all configuration management with environment variable fallbacks.
"""
import os
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class FirebaseConfig:
    """Firebase configuration with validation."""
    project_id: str
    private_key_id: Optional[str] = None
    private_key: Optional[str] = None
    client_email: Optional[str] = None
    
    def __post_init__(self):
        """Validate Firebase configuration."""
        if not self.project_id:
            raise ValueError("Firebase project_id is required")
        if self.private_key and not self.client_email:
            raise ValueError("Client email required when using private key")

@dataclass
class TradingConfig:
    """Trading-specific configuration."""
    simulation_mode: bool = True
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%

@dataclass
class LLMConfig:
    """LLM configuration with fallback to local models."""
    provider: str = "openai"  # openai, anthropic, local
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    @property
    def api_key(self) -> Optional[str]:
        """Safely retrieve API key from environment."""
        key_name = f"{self.provider.upper()}_API_KEY"
        key = os.getenv(key_name)
        if not key and self.provider != "local":
            logger.warning(f"Missing API key for {self.provider}")
        return key

class Config:
    """Main configuration singleton."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize configuration from environment."""
        self.firebase = FirebaseConfig(
            project_id=os.getenv("FIREBASE_PROJECT_ID", "autonomous-trading-dev"),
            private_key_id=os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            private_key=os.getenv("FIREBASE_PRIVATE_KEY"),
            client_email=os.getenv("FIREBASE_CLIENT_EMAIL")
        )
        
        self.trading = TradingConfig(
            simulation_mode=os.getenv("TRADING_SIMULATION", "True").lower() == "true",
            initial_capital=float(os.getenv("INITIAL_CAPITAL", "100000")),
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.1")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.02")),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.04"))
        )
        
        self.llm = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model_name=os.getenv("LLM_MODEL", "gpt-4-turbo-preview"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000"))
        )
        
        # Log configuration status
        logger.info(f"Configuration loaded: Simulation={self.trading.simulation_mode}")