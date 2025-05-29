"""
Main controller for QuantEvolve
"""

import asyncio
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from quantevolve.config import Config, EvaluatorConfig, load_config
from quantevolve.database import Program, ProgramDatabase
from quantevolve.evaluator import Evaluator
from quantevolve.llm.ensemble import LLMEnsemble
from quantevolve.prompt.sampler import PromptSampler
from quantevolve.utils.code_utils import (
    apply_diff,
    extract_code_language,
    extract_diffs,
    format_diff_summary,
    parse_evolve_blocks,
    parse_full_rewrite,
)

logger = logging.getLogger(__name__)


class QuantEvolveController:
    """
    Main controller for QuantEvolve

    Orchestrates the evolution process, coordinating between the prompt sampler,
    LLM ensemble, evaluator, and program database.

    Features:
    - Tracks the absolute best program across evolution steps
    - Ensures the best solution is not lost during the MAP-Elites process
    - Always includes the best program in the selection process for inspiration
    - Maintains detailed logs and metadata about improvements
    """

    def __init__(
        self,
        initial_program_path: str = "quantevolve/strategy/initial_strategy.py",
        evaluation_file: str = "quantevolve/evaluation/quant_evaluator.py",
        config_path: Optional[str] = None,  # Default handled by load_config
        config: Optional[Config] = None,
        output_dir: Optional[str] = "quantevolve_output",
    ):
        # Load configuration
        if config is not None:
            # Use provided Config object directly
            self.config = config
        else:
            # Load from file or use defaults
            self.config = load_config(config_path)  # config_path default will be handled here

        # Set up output directory
        # The output_dir argument now has a default in the signature.
        # If None is explicitly passed, it will use the default from the signature.
        # If a value is passed, it will use that value.
        self.output_dir = output_dir if output_dir is not None else "quantevolve_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        self._setup_logging()

        # Load initial program
        self.initial_program_path = (
            initial_program_path
            if initial_program_path is not None
            else "quantevolve/strategy/initial_strategy.py"
        )
        self.initial_program_code = self._load_initial_program()
        self.language = extract_code_language(self.initial_program_code)

        # Handle evaluation_file default
        self.evaluation_file = (
            evaluation_file
            if evaluation_file is not None
            else "quantevolve/evaluation/quant_evaluator.py"
        )

        # Extract file extension from initial program
        self.file_extension = os.path.splitext(self.initial_program_path)[1]
        if not self.file_extension:
            # Default to .py if no extension found
            self.file_extension = ".py"
        else:
            # Make sure it starts with a dot
            if not self.file_extension.startswith("."):
                self.file_extension = f".{self.file_extension}"

        # Initialize components
        self.llm_ensemble = LLMEnsemble(self.config.llm)
        self.prompt_sampler = PromptSampler(self.config.prompt)
        self.database = ProgramDatabase(self.config.database)
        self.evaluator = Evaluator(self.config.evaluator, self.evaluation_file, self.llm_ensemble)

        logger.info(
            f"Initialized QuantEvolveController with {initial_program_path} "
            f"and {self.evaluation_file}"
        )

    def _setup_logging(self) -> None:
        """Set up logging"""
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))

        # Add file handler
        log_file = os.path.join(log_dir, f"quantevolve_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(console_handler)

        logger.info(f"Logging to {log_file}")

    def _load_initial_program(self) -> str:
        """Load the initial program from file"""
        with open(self.initial_program_path, "r", encoding="utf-8") as f:
            return f.read()

    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
    ) -> Program:
        """
        Run the evolution process

        Args:
            iterations: Maximum number of iterations (uses config if None)
            target_score: Target score to reach (continues until reached if specified)

        Returns:
            Best strategy found
        """
        max_iterations = iterations or self.config.max_iterations

        # Define start_iteration before creating the initial program
        start_iteration = self.database.last_iteration

        # Only add initial program if starting fresh (not resuming from checkpoint)
        # Check if we're resuming AND no program matches initial code to avoid pollution
        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and not any(
                p.code == self.initial_program_code for p in self.database.programs.values()
            )
        )

        if should_add_initial:
            logger.info("Adding initial strategy to database")
            initial_program_id = str(uuid.uuid4())  # Keep 'program' for internal ID consistency

            # Evaluate the initial strategy
            initial_metrics = await self.evaluator.evaluate_program(
                self.initial_program_code, initial_program_id
            )

            initial_strategy_program = (
                Program(  # Keep Program class, but variable name can be more specific
                    id=initial_program_id,
                    code=self.initial_program_code,
                    language=self.language,
                    metrics=initial_metrics,
                    iteration_found=start_iteration,
                )
            )

            self.database.add(initial_strategy_program)
        else:
            logger.info(
                f"Skipping initial strategy addition (resuming from iteration {start_iteration} with {len(self.database.programs)} existing programs)"
            )

        # Main evolution loop
        total_iterations = start_iteration + max_iterations

        logger.info(
            f"Starting evolution from iteration {start_iteration} for {max_iterations} iterations (total: {total_iterations})"
        )

        for i in range(start_iteration, total_iterations):
            iteration_start = time.time()

            # Sample parent and inspirations
            parent, inspirations = self.database.sample()

            # Build prompt
            prompt = self.prompt_sampler.build_prompt(
                current_program=parent.code,
                parent_program=parent.code,  # We don't have the parent's code, use the same
                program_metrics=parent.metrics,
                previous_programs=[p.to_dict() for p in self.database.get_top_programs(3)],
                top_programs=[p.to_dict() for p in inspirations],
                language=self.language,
                evolution_round=i,
                allow_full_rewrite=self.config.allow_full_rewrites,
            )

            # Generate code modification
            try:
                llm_response = await self.llm_ensemble.generate_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                )

                # Parse the response
                if self.config.diff_based_evolution:
                    # First try to extract diff blocks
                    diff_blocks = extract_diffs(llm_response)

                    if diff_blocks:
                        # Apply the diffs if found
                        child_code = apply_diff(parent.code, llm_response)
                        changes_summary = format_diff_summary(diff_blocks)
                    else:
                        # If no diff blocks found, try to extract evolve blocks and apply changes
                        evolve_blocks = parse_evolve_blocks(parent.code)

                        if evolve_blocks and self.config.allow_full_rewrites:
                            # Extract full code from LLM response
                            new_code = parse_full_rewrite(llm_response, self.language)

                            if new_code:
                                # Apply changes only to evolve blocks
                                parent_lines = parent.code.split("\n")
                                new_code_lines = new_code.split("\n")

                                # Create a modified version of the code
                                child_code = parent.code

                                # Look for code blocks in LLM response
                                for start_line, end_line, block_content in evolve_blocks:
                                    # Try to find corresponding code in LLM response
                                    if (
                                        len(new_code_lines) > 10
                                    ):  # Ensure there's enough code to work with
                                        # Replace the evolve block with new code
                                        child_code = child_code.replace(
                                            "# EVOLVE-BLOCK-START\n{}\n".format(block_content),
                                            "# EVOLVE-BLOCK-START\n{}\n".format(new_code),
                                        )

                                changes_summary = "Updated EVOLVE-BLOCK sections"
                            else:
                                logger.warning(
                                    "Iteration {}: No valid code found in response for evolve blocks".format(
                                        i + 1
                                    )
                                )
                                continue
                        else:
                            logger.warning(
                                "Iteration {}: No valid diffs found in response and no evolve blocks to update".format(
                                    i + 1
                                )
                            )
                            continue
                else:
                    # Parse full rewrite
                    new_code = parse_full_rewrite(llm_response, self.language)

                    if not new_code:
                        logger.warning(f"Iteration {i+1}: No valid code found in response")
                        continue

                    child_code = new_code
                    changes_summary = "Full rewrite"

                # Check code length
                if len(child_code) > self.config.max_code_length:
                    logger.warning(
                        f"Iteration {i+1}: Generated code exceeds maximum length "
                        f"({len(child_code)} > {self.config.max_code_length})"
                    )
                    continue

                # Evaluate the child program
                child_id = str(uuid.uuid4())
                child_metrics = await self.evaluator.evaluate_program(child_code, child_id)

                # Create a child program
                child_program = Program(
                    id=child_id,
                    code=child_code,
                    language=self.language,
                    parent_id=parent.id,
                    generation=parent.generation + 1,
                    metrics=child_metrics,
                    metadata={
                        "changes": changes_summary,
                        "parent_metrics": parent.metrics,
                    },
                )

                # Add to database
                self.database.add(child_program, iteration=i + 1)

                # Log progress
                iteration_time = time.time() - iteration_start
                self._log_iteration(i, parent, child_program, iteration_time)

                # Specifically check if this is the new best program
                if self.database.best_program_id == child_program.id:
                    logger.info(
                        "*** New best strategy found at iteration {}: {}".format(
                            i + 1, child_program.id
                        )
                    )
                    metrics_str = ", ".join(
                        [
                            "{0}={1:.4f}".format(name, value)
                            for name, value in child_program.metrics.items()
                        ]
                    )
                    logger.info("Metrics for new best strategy: {}".format(metrics_str))

                # Save checkpoint
                if (i + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(i + 1)

                # Check if target score reached
                if target_score is not None:
                    avg_score = sum(child_metrics.values()) / max(1, len(child_metrics))
                    if avg_score >= target_score:
                        logger.info(f"Target score {target_score} reached after {i+1} iterations")
                        break

            except Exception as e:
                logger.error(f"Error in iteration {i+1}: {str(e)}")
                continue

        # Get the best program using our tracking mechanism
        best_strategy = None  # Renamed variable
        if self.database.best_program_id:  # Internal database ID still 'program'
            best_strategy = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best strategy: {self.database.best_program_id}")

        # Fallback to calculating best strategy if tracked strategy not found
        if best_strategy is None:
            best_strategy = self.database.get_best_program()  # Method name in database class
            logger.info("Using calculated best strategy (tracked strategy not found)")

        # Check if there's a better strategy by combined_score that wasn't tracked
        if (
            best_strategy and "combined_score" in best_strategy.metrics
        ):  # Check if best_strategy is not None
            best_by_combined = self.database.get_best_program(metric="combined_score")
            if (
                best_by_combined
                and best_by_combined.id != best_strategy.id
                and "combined_score" in best_by_combined.metrics
            ):
                # If the combined_score of this strategy is significantly better, use it instead
                if (
                    best_by_combined.metrics["combined_score"]
                    > best_strategy.metrics["combined_score"] + 0.02
                ):
                    logger.warning(
                        f"Found strategy with better combined_score: {best_by_combined.id}"
                    )
                    logger.warning(
                        f"Score difference for strategy {best_strategy.id} ({best_strategy.metrics['combined_score']:.4f}) vs {best_by_combined.id} ({best_by_combined.metrics['combined_score']:.4f})"
                    )
                    best_strategy = best_by_combined
        elif not best_strategy:  # Handle case where no strategy was found at all
            logger.warning("No best strategy could be determined.")

        if best_strategy:
            logger.info(
                f"Evolution complete. Best strategy ({best_strategy.id}) has metrics: "
                f"{', '.join(f'{name}={value:.4f}' for name, value in best_strategy.metrics.items())}"
            )

            # Save the best strategy (using our tracked best strategy)
            self._save_best_program(best_strategy)  # Pass the strategy to save

            return best_strategy
        else:
            logger.warning("No valid strategies found during evolution")
            # Return None if no strategies found
            return None

    def _log_iteration(
        self,
        iteration: int,
        parent: Program,
        child: Program,
        elapsed_time: float,
    ) -> None:
        """
        Log iteration progress

        Args:
            iteration: Iteration number
            parent: Parent program
            child: Child program
            elapsed_time: Elapsed time in seconds
        """

        def _safe_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        # Build numeric metrics dict for child
        numeric_metrics = {}
        for metric, value in child.metrics.items():
            num_val = _safe_float(value)
            if num_val is not None:
                numeric_metrics[metric] = num_val

        # Compute improvement over parent metrics
        improvement = {}
        for metric, num_val in numeric_metrics.items():
            parent_val = _safe_float(parent.metrics.get(metric))
            if parent_val is not None:
                improvement[metric] = num_val - parent_val

        # Format metrics and improvements
        metrics_str = ", ".join(f"{name}={val:.4f}" for name, val in numeric_metrics.items())
        improvement_str = ", ".join(f"{name}={diff:+.4f}" for name, diff in improvement.items())

        logger.info(
            f"Iteration {iteration+1}: Child {child.id} from parent {parent.id} "
            f"in {elapsed_time:.2f}s. Metrics: {metrics_str} (Δ: {improvement_str})"
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """
        Save a checkpoint

        Args:
            iteration: Current iteration number
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create specific checkpoint directory
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save the database
        self.database.save(checkpoint_path, iteration)

        # Save the best strategy found so far
        best_strategy_at_checkpoint = None  # Renamed variable
        if self.database.best_program_id:  # Internal database ID still 'program'
            best_strategy_at_checkpoint = self.database.get(self.database.best_program_id)
        else:
            best_strategy_at_checkpoint = (
                self.database.get_best_program()
            )  # Method name in database class

        if best_strategy_at_checkpoint:
            # Save the best strategy at this checkpoint
            best_strategy_path = os.path.join(
                checkpoint_path, f"best_strategy{self.file_extension}"
            )
            with open(best_strategy_path, "w") as f:
                f.write(best_strategy_at_checkpoint.code)

            # Save metrics
            best_strategy_info_path = os.path.join(checkpoint_path, "best_strategy_info.json")
            with open(best_strategy_info_path, "w") as f:
                import json

                json.dump(
                    {
                        "id": best_strategy_at_checkpoint.id,
                        "generation": best_strategy_at_checkpoint.generation,
                        "iteration": best_strategy_at_checkpoint.iteration_found,
                        "current_iteration": iteration,
                        "metrics": best_strategy_at_checkpoint.metrics,
                        "language": best_strategy_at_checkpoint.language,
                        "timestamp": best_strategy_at_checkpoint.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Saved best strategy at checkpoint {iteration} ({best_strategy_at_checkpoint.id}) with metrics: "
                f"{', '.join(f'{name}={value:.4f}' for name, value in best_strategy_at_checkpoint.metrics.items())}"
            )

        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

    def _save_best_program(
        self, strategy_to_save: Optional[Program] = None
    ) -> None:  # Parameter renamed for clarity
        """
        Save the best strategy

        Args:
            strategy_to_save: Best strategy (if None, uses the tracked best strategy)
        """
        # If no strategy is provided, use the tracked best strategy from the database
        if strategy_to_save is None:
            if self.database.best_program_id:  # Internal database ID still 'program'
                strategy_to_save = self.database.get(self.database.best_program_id)
            else:
                # Fallback to calculating best strategy if no tracked best strategy
                strategy_to_save = self.database.get_best_program()  # Method name in database class

        if not strategy_to_save:
            logger.warning("No best strategy found to save")
            return

        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Use the extension from the initial program file
        filename = f"best_strategy{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(strategy_to_save.code)

        # Save complete strategy info including metrics
        info_path = os.path.join(best_dir, "best_strategy_info.json")
        with open(info_path, "w") as f:
            import json

            json.dump(
                {
                    "id": strategy_to_save.id,
                    "generation": strategy_to_save.generation,
                    "iteration": strategy_to_save.iteration_found,
                    "timestamp": strategy_to_save.timestamp,
                    "parent_id": strategy_to_save.parent_id,
                    "metrics": strategy_to_save.metrics,
                    "language": strategy_to_save.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(
            f"Saved best strategy ({strategy_to_save.id}) to {code_path} with strategy info to {info_path}"
        )
