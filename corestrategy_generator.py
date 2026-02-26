"""
Autonomous Strategy Generator using LLMs.
Generates, validates, and optimizes trading strategies based on market conditions.
"""
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np
from firebase_admin import firestore
from pydantic import BaseModel, ValidationError

from config import Config

logger = logging.getLogger(__name__)

class StrategySpecification(BaseModel):
    """Pydantic model for strategy validation."""
    name: str
    description: str
    asset_class: str  # forex, crypto, equities, etc.
    timeframe: str  # 1m, 5m, 1h, 1d
    indicators: List[str]
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_parameters: Dict[str, float]
    generated_at: datetime
    version: str = "1.0.0"
    
    class Config:
        arbitrary_types_allowed = True

class StrategyGenerator:
    """Main strategy generation class using LLM."""
    
    def __init__(self):
        self.config = Config()
        self.db = firestore.client()
        self.strategies_ref = self.db.collection('strategies')
        
    def _build_prompt(self, market_data: pd.DataFrame, market_condition: str) -> str:
        """Build LLM prompt with market context."""
        
        prompt = f"""You are an expert quantitative trading strategist. Generate a novel trading strategy based on current market conditions.

MARKET CONTEXT:
- Current condition: {market_condition}
- Recent volatility: {market_data['close'].pct_change().std():.4f}
- Trend direction: {'Bullish' if market_data['close'].iloc[-1] > market_data['close'].iloc[-20] else 'Bearish'}
- Timeframe: {self.config.trading.timeframe if hasattr