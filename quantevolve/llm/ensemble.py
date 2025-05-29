"""
Model ensemble for LLMs
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple

from quantevolve.config import LLMConfig
from quantevolve.llm.base import LLMInterface
from quantevolve.llm.gemini import GeminiLLM

logger = logging.getLogger(__name__)


class LLMEnsemble:
    """Ensemble of LLMs for generating diverse code modifications"""

    def __init__(self, config: LLMConfig):
        self.config = config

        # Initialize primary and secondary models
        primary_model_name = config.primary_model
        secondary_model_name = config.secondary_model

        # Always use GeminiLLM
        logger.info("LLMEnsemble: Using GeminiLLM for all models.")
        LLMClientClass = GeminiLLM

        self.primary_model = LLMClientClass(config, model=primary_model_name)
        if secondary_model_name:  # Ensure secondary_model is not None or empty
            self.secondary_model = LLMClientClass(config, model=secondary_model_name)
            # Model weights for sampling
            self._weights = [
                config.primary_model_weight,
                config.secondary_model_weight,
            ]
            # Normalize weights
            total = sum(self._weights)
            if total > 0:
                self._weights = [w / total for w in self._weights]
            else:  # Avoid division by zero if weights are zero
                self._weights = [0.5, 0.5]  # Default to equal weighting

            logger.info(
                f"Initialized LLM ensemble with models: "
                f"{primary_model_name} (weight: {self._weights[0]:.2f}), "
                f"{secondary_model_name} (weight: {self._weights[1]:.2f})"
            )
        else:
            self.secondary_model = None
            self._weights = [1.0]  # Only primary model
            logger.info(
                f"Initialized LLM ensemble with primary model: "
                f"{primary_model_name} (weight: 1.00)"
            )

        # Model weights for sampling
        self._weights = [
            config.primary_model_weight,
            config.secondary_model_weight,
        ]

        # Normalize weights
        total = sum(self._weights)
        self._weights = [w / total for w in self._weights]

        logger.info(
            f"Initialized LLM ensemble with models: "
            f"{config.primary_model} (weight: {self._weights[0]:.2f}), "
            f"{config.secondary_model} (weight: {self._weights[1]:.2f})"
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using a randomly selected model based on weights"""
        model = self._sample_model()
        return await model.generate(prompt, **kwargs)

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        model = self._sample_model()
        return await model.generate_with_context(system_message, messages, **kwargs)

    def _sample_model(self) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        models = [self.primary_model]
        if self.secondary_model:
            models.append(self.secondary_model)

        if not models:  # Should not happen if constructor is correct
            raise ValueError("No models available in LLMEnsemble.")

        if len(models) == 1:
            return models[0]

        index = random.choices(range(len(models)), weights=self._weights[: len(models)], k=1)[0]
        return models[index]

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
