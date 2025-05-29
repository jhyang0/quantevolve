"""
Command-line interface for QuantEvolve
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

from quantevolve.controller import QuantEvolveController # Changed import
from quantevolve.config import Config, load_config # Changed import

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="QuantEvolve - Evolutionary Quantitative Trading Strategy Development")

    # Arguments that override QuantEvolveController defaults
    parser.add_argument(
        "--initial_strategy",
        help="Path to the initial strategy file (overrides default: quantevolve/strategy/initial_strategy.py)",
        default=None 
    )
    parser.add_argument(
        "--evaluator_file",
        help="Path to the evaluator file (overrides default: quantevolve/evaluation/quant_evaluator.py)",
        default=None
    )
    parser.add_argument(
        "--config", 
        "-c", 
        help="Path to configuration file (YAML) (overrides default: configs/quantevolve_config.yaml or configs/quantevolve_default_config.yaml)", 
        default=None
    )
    parser.add_argument(
        "--output", 
        "-o", 
        help="Output directory for results (overrides default: quantevolve_output/)", 
        default=None
    )

    # General arguments
    parser.add_argument(
        "--iterations", "-i", help="Maximum number of evolution iterations", type=int, default=None
    )

    parser.add_argument(
        "--target-score", "-t", help="Target score to reach", type=float, default=None
    )

    parser.add_argument(
        "--log-level",
        "-l",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint directory to resume from (e.g., quantevolve_output/checkpoints/checkpoint_50)",
        default=None,
    )

    parser.add_argument("--api-base", help="Base URL for the LLM API", default=None)

    parser.add_argument("--primary-model", help="Primary LLM model name", default=None)

    parser.add_argument("--secondary-model", help="Secondary LLM model name", default=None)

    return parser.parse_args()


async def main_async() -> int:
    """
    Main asynchronous entry point

    Returns:
        Exit code
    """
    args = parse_args()

    # Create config object first, as QuantEvolveController needs it.
    # Command-line LLM args override config file.
    loaded_config = load_config(args.config) # args.config can be None, load_config handles it

    if args.api_base:
        loaded_config.llm.api_base = args.api_base
        logger.info(f"Overriding LLM API base from CLI: {loaded_config.llm.api_base}")
    if args.primary_model:
        loaded_config.llm.primary_model = args.primary_model
        logger.info(f"Overriding primary LLM model from CLI: {loaded_config.llm.primary_model}")
    if args.secondary_model:
        loaded_config.llm.secondary_model = args.secondary_model
        logger.info(f"Overriding secondary LLM model from CLI: {loaded_config.llm.secondary_model}")
    
    # The controller will use its defaults if args are None
    initial_strategy_path = args.initial_strategy
    evaluator_path = args.evaluator_file
    output_path = args.output

    # Check existence only if paths are explicitly provided via CLI
    if initial_strategy_path and not os.path.exists(initial_strategy_path):
        print(f"Error: Initial strategy file '{initial_strategy_path}' not found.")
        return 1
    if evaluator_path and not os.path.exists(evaluator_path):
        print(f"Error: Evaluator file '{evaluator_path}' not found.")
        return 1

    # Initialize QuantEvolveController
    try:
        # Pass args directly. If they are None, controller uses its defaults.
        # Pass loaded_config as the 'config' object.
        # config_path is effectively handled by load_config already, so not strictly needed by controller if config object is passed.
        controller = QuantEvolveController(
            initial_program_path=initial_strategy_path, # Name in controller is initial_program_path
            evaluation_file=evaluator_path,
            config=loaded_config, # Pass the potentially modified config object
            output_dir=output_path,
        )

        # Load from checkpoint if specified
        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint directory '{args.checkpoint}' not found")
                return 1
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            controller.database.load(args.checkpoint)
            logger.info(
                f"Checkpoint loaded successfully (resuming from iteration {controller.database.last_iteration})"
            )

        # Override log level if specified (this was already here and is fine)
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))
            logger.info(f"Log level set to {args.log_level} from CLI.")


        # Run evolution
        best_strategy = await controller.run(
            iterations=args.iterations,
            target_score=args.target_score,
        )

        if not best_strategy:
            print("Evolution finished, but no best strategy was determined.")
            return 1

        # Get the checkpoint path
        checkpoint_dir = os.path.join(controller.output_dir, "checkpoints")
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [
                os.path.join(checkpoint_dir, d)
                for d in os.listdir(checkpoint_dir)
                if os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            if checkpoints:
                latest_checkpoint = sorted(
                    checkpoints, key=lambda x: int(x.split("_")[-1]) if "_" in x else 0
                )[-1]

        print(f"\nEvolution complete!")
        print(f"Best strategy ({best_strategy.id}) metrics:")
        for name, value in best_strategy.metrics.items():
            print(f"  {name}: {value:.4f}")
        
        best_strategy_path = os.path.join(controller.output_dir, "best", f"best_strategy{controller.file_extension}")
        print(f"Best strategy saved to: {best_strategy_path}")

        if latest_checkpoint:
            print(f"\nLatest checkpoint saved at: {latest_checkpoint}")
            print(f"To resume, use: --checkpoint {latest_checkpoint}")
        
        return 0

    except Exception as e:
        logger.error(f"An error occurred during QuantEvolve execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """
    Main entry point

    Returns:
        Exit code
    """
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
