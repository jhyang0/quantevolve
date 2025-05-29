"""
Prompt sampling for QuantEvolve

This module provides tools for generating prompts for the LLM to evolve
quantitative trading strategies based on performance metrics.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from quantevolve.config import PromptConfig
from quantevolve.prompt.templates import TemplateManager

logger = logging.getLogger(__name__)


class PromptSampler:
    """Generates prompts for quantitative trading strategy evolution"""

    def __init__(self, config: PromptConfig):
        self.config = config
        self.template_manager = TemplateManager(config.template_dir)

        # Initialize the random number generator
        random.seed()

        # Store custom template mappings
        self.system_template_override = None
        self.user_template_override = None

        logger.info("Initialized prompt sampler")

    def set_templates(
        self, system_template: Optional[str] = None, user_template: Optional[str] = None
    ) -> None:
        """
        Set custom templates to use for this sampler

        Args:
            system_template: Template name for system message
            user_template: Template name for user message
        """
        self.system_template_override = system_template
        self.user_template_override = user_template
        logger.info(f"Set custom templates: system={system_template}, user={user_template}")

    def build_prompt(
        self,
        current_program: str,
        parent_program: str,
        program_metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        language: str = "python",
        evolution_round: int = 0,
        allow_full_rewrite: bool = False,
        template_key: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Build a prompt for the LLM to evolve quantitative trading strategies

        Args:
            current_program: Current trading strategy code
            parent_program: Parent strategy from which current was derived
            program_metrics: Dictionary of performance metrics (e.g., Sharpe ratio, PnL, max drawdown)
            previous_programs: List of previous strategy attempts
            top_programs: List of top-performing trading strategies
            language: Programming language (typically Python)
            evolution_round: Current evolution round
            allow_full_rewrite: Whether to allow a full strategy rewrite
            template_key: Optional override for template key

        Returns:
            Dictionary with 'system' and 'user' keys
        """
        # Select template based on whether we want a full rewrite (with overrides)
        if template_key:
            # Use explicitly provided template key
            user_template_key = template_key
        elif self.user_template_override:
            # Use the override set with set_templates
            user_template_key = self.user_template_override
        else:
            # Default behavior
            user_template_key = "full_rewrite_user" if allow_full_rewrite else "diff_user"

        # Get the template
        user_template = self.template_manager.get_template(user_template_key)

        # Use system template override if set
        if self.system_template_override:
            system_message = self.template_manager.get_template(self.system_template_override)
        else:
            system_message = self.config.system_message
            # If system_message is a template name rather than content, get the template
            if system_message in self.template_manager.templates:
                system_message = self.template_manager.get_template(system_message)

        # Format metrics
        metrics_str = self._format_metrics(program_metrics)

        # Identify areas for improvement
        improvement_areas = self._identify_improvement_areas(
            current_program, parent_program, program_metrics, previous_programs
        )

        # Format evolution history
        evolution_history = self._format_evolution_history(
            previous_programs, top_programs, language
        )

        # Apply stochastic template variations if enabled
        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        # Format the final user message
        user_message = user_template.format(
            metrics=metrics_str,
            improvement_areas=improvement_areas,
            evolution_history=evolution_history,
            current_program=current_program,
            language=language,
        )

        return {
            "system": system_message,
            "user": user_message,
        }

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format trading performance metrics for the prompt"""
        if not metrics:
            return "No metrics available"

        # Format each metric with proper handling for string values
        formatted_metrics = []
        for name, value in metrics.items():
            try:
                formatted_metrics.append(f"{name}: {float(value):.4f}")
            except (ValueError, TypeError):
                formatted_metrics.append(f"{name}: {str(value)}")
                
        return ", ".join(formatted_metrics)

    def _identify_improvement_areas(
        self,
        current_program: str,
        parent_program: str,
        metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
    ) -> str:
        """Identify potential areas for improvement in trading strategies"""
        # This method could be expanded to include more sophisticated analysis
        # For now, we'll use a simple approach

        improvement_areas = []

        # Check program length
        if len(current_program) > 500:
            improvement_areas.append(
                "Consider simplifying the code to improve readability and maintainability"
            )

        # Check for performance patterns in previous attempts
        if len(previous_programs) >= 2:
            recent_attempts = previous_programs[-2:]
            metrics_improved = []
            metrics_regressed = []

            for metric, value in metrics.items():
                improved = True
                regressed = True

                for attempt in recent_attempts:
                    if attempt["metrics"].get(metric, 0) <= value:
                        regressed = False
                    if attempt["metrics"].get(metric, 0) >= value:
                        improved = False

                if improved and metric not in metrics_improved:
                    metrics_improved.append(metric)
                if regressed and metric not in metrics_regressed:
                    metrics_regressed.append(metric)

            if metrics_improved:
                improvement_areas.append(
                    f"Metrics showing improvement: {', '.join(metrics_improved)}. "
                    "Consider continuing with similar changes."
                )

            if metrics_regressed:
                improvement_areas.append(
                    f"Metrics showing regression: {', '.join(metrics_regressed)}. "
                    "Consider reverting or revising recent changes in these areas."
                )

        # If we don't have specific improvements to suggest
        if not improvement_areas:
            improvement_areas.append(
                "Focus on optimizing the code for better performance on the target metrics"
            )

        return "\n".join([f"- {area}" for area in improvement_areas])

    def _format_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """Format the evolution history for the prompt"""
        # Get templates
        history_template = self.template_manager.get_template("evolution_history")
        previous_attempt_template = self.template_manager.get_template("previous_attempt")
        top_program_template = self.template_manager.get_template("top_program")

        # Format previous attempts (most recent first)
        previous_attempts_str = ""
        selected_previous = previous_programs[-min(3, len(previous_programs)) :]

        for i, program in enumerate(reversed(selected_previous)):
            attempt_number = len(previous_programs) - i
            changes = program.get("changes", "Unknown changes")

            # Format performance metrics
            metric_strs = []
            for name, value in program.get("metrics", {}).items():
                try:
                    metric_strs.append(f"{name}: {float(value):.4f}")
                except (ValueError, TypeError):
                    metric_strs.append(f"{name}: {str(value)}")
            performance_str = ", ".join(metric_strs)

            # Determine outcome based on comparison with parent
            parent_metrics = program.get("parent_metrics", {})
            outcome = "Mixed results"

            def _safe_float(val):
                """Safely convert a value to float, returning None if conversion fails."""
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None

            # Build list of numeric metric pairs (program_value, parent_value)
            numeric_pairs = [
                (_safe_float(program.get("metrics", {}).get(m)), _safe_float(parent_metrics.get(m)))
                for m in program.get("metrics", {})
            ]
            # Filter out any non-numeric comparisons
            numeric_pairs = [(p, pa) for p, pa in numeric_pairs if p is not None and pa is not None]

            if numeric_pairs and all(p >= pa for p, pa in numeric_pairs):
                outcome = "Improvement in all metrics"
            elif numeric_pairs and all(p <= pa for p, pa in numeric_pairs):
                outcome = "Regression in all metrics"

            previous_attempts_str += (
                previous_attempt_template.format(
                    attempt_number=attempt_number,
                    changes=changes,
                    performance=performance_str,
                    outcome=outcome,
                )
                + "\n\n"
            )

        # Format top programs
        top_programs_str = ""
        selected_top = top_programs[: min(self.config.num_top_programs, len(top_programs))]

        for i, program in enumerate(selected_top):
            # Extract a snippet (first 10 lines) for display
            program_code = program.get("code", "")
            program_snippet = "\n".join(program_code.split("\n")[:10])
            if len(program_code.split("\n")) > 10:
                program_snippet += "\n# ... (truncated for brevity)"

            # Calculate a composite score for the trading strategy
            # For trading strategies, we might weight certain metrics more heavily
            # For example, Sharpe ratio and drawdown might be more important than raw returns
            metrics_dict = program.get("metrics", {})
            try:
                # Try to calculate weighted score if numeric values are available
                score = 0
                count = 0
                for name, value in metrics_dict.items():
                    try:
                        float_value = float(value)
                        # Give higher weight to key trading metrics
                        weight = 2.0 if name.lower() in ["sharpe", "sortino", "calmar"] else 1.0
                        score += float_value * weight
                        count += weight
                    except (ValueError, TypeError):
                        continue
                
                score = score / max(1, count)
            except Exception:
                # Fallback to simple average
                score = sum([float(v) if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit()) 
                             else 0 for v in metrics_dict.values()]) / max(1, len(metrics_dict))

            # Extract key features of successful trading strategies
            key_features = program.get("key_features", [])
            if not key_features:
                key_features = []
                for name, value in program.get("metrics", {}).items():
                    try:
                        key_features.append(f"Performs well on {name} ({float(value):.4f})")
                    except (ValueError, TypeError):
                        key_features.append(f"Performs well on {name} ({str(value)})")
                
                # If no metrics were found, add a generic message
                if not key_features:
                    key_features = ["Notable trading strategy"]

            key_features_str = ", ".join(key_features)

            top_programs_str += (
                top_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    language=language,
                    program_snippet=program_snippet,
                    key_features=key_features_str,
                )
                + "\n\n"
            )

        # Combine into full history
        return history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=top_programs_str.strip(),
        )

    def _apply_template_variations(self, template: str) -> str:
        """Apply stochastic variations to the template"""
        result = template

        # Apply variations defined in the config
        for key, variations in self.config.template_variations.items():
            if variations and f"{{{key}}}" in result:
                chosen_variation = random.choice(variations)
                result = result.replace(f"{{{key}}}", chosen_variation)

        return result
