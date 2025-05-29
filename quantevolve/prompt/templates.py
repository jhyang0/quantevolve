"""
Prompt templates for QuantEvolve

This module provides template definitions for generating prompts
to guide the LLM in evolving quantitative trading strategies.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Base system message template for trading strategy evolution
BASE_SYSTEM_TEMPLATE = """You are an expert quantitative trader and algorithm developer tasked with iteratively improving trading strategies.
Your job is to analyze the current trading strategy and suggest improvements based on performance feedback from previous versions.
Focus on making targeted changes that will improve financial metrics such as Sharpe ratio, profit and loss (PnL), maximum drawdown, and risk-adjusted returns.
You should understand concepts like technical indicators, risk management, position sizing, market regimes, and alpha generation.
"""

# User message template for diff-based trading strategy evolution
DIFF_USER_TEMPLATE = """# Current Trading Strategy Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

# Strategy Evolution History
{evolution_history}

# Current Trading Strategy
```{language}
{current_program}
```

# Task
Suggest improvements to the trading strategy that will lead to better financial performance on the specified metrics.
Consider factors such as:
- Signal generation and entry/exit conditions
- Risk management and position sizing
- Handling of market regimes and volatility
- Computational efficiency and robustness

IMPORTANT: Pay close attention to '# EVOLVE-BLOCK-START' and '# EVOLVE-BLOCK-END' markers in the code. These markers indicate sections that should be modified or improved. Focus your changes primarily on the code between these markers.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
"""

# User message template for full trading strategy rewrite
FULL_REWRITE_USER_TEMPLATE = """# Current Trading Strategy Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

# Strategy Evolution History
{evolution_history}

# Current Trading Strategy
```{language}
{current_program}
```

# Task
Rewrite the trading strategy to improve its financial performance on the specified metrics.
Provide the complete new strategy code.

IMPORTANT: Pay close attention to '# EVOLVE-BLOCK-START' and '# EVOLVE-BLOCK-END' markers in the code. These markers indicate sections that should be completely rewritten. If you rewrite the entire strategy, make sure to preserve these markers but improve the code between them.

Consider these aspects in your rewrite:
- More effective signal generation and market timing
- Better risk management techniques
- Improved position sizing algorithms
- Adaptability to different market conditions
- Computational efficiency and robustness

IMPORTANT: Make sure your rewritten strategy maintains the same input data format and output trading signals
as the original strategy, but with improved internal implementation.

```{language}
# Your rewritten trading strategy here
```
"""

# Template for formatting trading strategy evolution history
EVOLUTION_HISTORY_TEMPLATE = """## Previous Strategy Versions

{previous_attempts}

## Top Performing Trading Strategies

{top_programs}
"""

# Template for formatting a previous strategy version
PREVIOUS_ATTEMPT_TEMPLATE = """### Strategy Version {attempt_number}
- Changes: {changes}
- Performance Metrics: {performance}
- Trading Outcome: {outcome}
"""

# Template for formatting a top trading strategy
TOP_PROGRAM_TEMPLATE = """### Strategy {program_number} (Score: {score})
```{language}
{program_snippet}
```
Key trading features: {key_features}
"""

# Default templates dictionary
DEFAULT_TEMPLATES = {
    "system_message": BASE_SYSTEM_TEMPLATE,
    "diff_user": DIFF_USER_TEMPLATE,
    "full_rewrite_user": FULL_REWRITE_USER_TEMPLATE,
    "evolution_history": EVOLUTION_HISTORY_TEMPLATE,
    "previous_attempt": PREVIOUS_ATTEMPT_TEMPLATE,
    "top_program": TOP_PROGRAM_TEMPLATE,
}


class TemplateManager:
    """Manages templates for trading strategy prompt generation"""

    def __init__(self, template_dir: Optional[str] = None):
        self.templates = DEFAULT_TEMPLATES.copy()

        # Load templates from directory if provided
        if template_dir and os.path.isdir(template_dir):
            self._load_templates_from_dir(template_dir)

    def _load_templates_from_dir(self, template_dir: str) -> None:
        """Load templates from a directory"""
        for file_path in Path(template_dir).glob("*.txt"):
            template_name = file_path.stem
            with open(file_path, "r") as f:
                self.templates[template_name] = f.read()

    def get_template(self, template_name: str) -> str:
        """Get a template by name"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self.templates[template_name]

    def add_template(self, template_name: str, template: str) -> None:
        """Add or update a template"""
        self.templates[template_name] = template
