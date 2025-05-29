"""
Google Gemini API interface for LLMs
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from quantevolve.llm.base import LLMInterface
from quantevolve.config import LLMConfig

logger = logging.getLogger(__name__)

class GeminiLLM(LLMInterface):
    """LLM interface using Google Gemini APIs"""

    def __init__(
        self,
        config: LLMConfig,
        model: Optional[str] = None,
    ):
        self.config = config
        self.model_name = model or config.primary_model
        self.api_key = config.api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY environment variable or api_key in config.")

        genai.configure(api_key=self.api_key)
        
        # Default safety settings - can be made configurable if needed
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings=self.safety_settings
        )

        logger.info(f"Initialized Gemini LLM with model: {self.model_name}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        # For Gemini, simple prompts are treated as a single user message.
        # System message can be part of the prompt or handled by generate_with_context.
        return await self.generate_with_context(
            system_message=self.config.system_message, # Use default system message from config
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        
        generation_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            # candidate_count=1, # Default is 1
        )

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "assistant":
                role = "model"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

        # Prepend system message as a special user message if provided
        if system_message and system_message.strip():
            # Insert system message at the beginning of conversation
            gemini_messages.insert(0, {"role": "user", "parts": [f"SYSTEM: {system_message}"]})

        retries = kwargs.get("retries", self.config.retries)
        retry_delay = kwargs.get("retry_delay", self.config.retry_delay)
        timeout_seconds = kwargs.get("timeout", self.config.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.model.generate_content_async(
                        contents=gemini_messages,
                        generation_config=generation_config,
                        safety_settings=self.safety_settings
                    ),
                    timeout=timeout_seconds
                )
                # Ensure there's content and parts before accessing
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text
                else:
                    # Handle cases where the response might be empty or blocked
                    logger.warning(f"Gemini response was empty or blocked. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}")
                    # You might want to return a specific string or raise an error here
                    # For now, returning empty string if blocked or no content
                    if response.candidates and response.candidates[0].finish_reason != 1: # 1 is STOP
                        return f"Content generation stopped due to: {response.candidates[0].finish_reason}"
                    return "" # Or raise an error

            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Gemini API call timeout on attempt {attempt + 1}/{retries + 1}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} Gemini API call attempts failed with timeout.")
                    raise
            except Exception as e:
                # Specific error handling for common Gemini issues can be added here
                # e.g., if isinstance(e, google.api_core.exceptions.PermissionDenied): ...
                if attempt < retries:
                    logger.warning(
                        f"Gemini API call error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} Gemini API call attempts failed with error: {str(e)}")
                    raise
        return "" # Should not be reached if retries are exhausted and an exception is raised
