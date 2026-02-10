# Part 1: AI-Assisted Coding with Claude Code

## Overview

In this lab you will use Claude Code to build a complete ML pipeline on the UCI Wine dataset. You will not write Python code yourself. Instead, you will use Claude Code's planning, coding, and quality-assurance features to have Claude build the pipeline for you. The lab is designed so that a single task naturally exercises every major Claude Code capability -- CLAUDE.md project rules, hooks for automated quality checks, skills for reusable workflows, slash commands for structured planning, and subagents for task decomposition.

The `demo/` folder contains a reference implementation built on the California Housing dataset (regression). Your task uses a **different dataset and problem type** (Wine classification), so you will need to rely on Claude Code rather than copying from the demo.

## What You Will Learn

- How **CLAUDE.md** enforces coding standards without manual review
- How **hooks** guarantee that ruff, py_compile, and test scaffolding run on every file write
- How the **`/plan` slash command** structures work before coding begins
- How **skills** (`/analyze-data`, `/evaluate-model`, `/generate-report`) provide reusable expert workflows
- How **subagents** decompose complex tasks into focused subtasks

## Prerequisites

- Claude Code CLI installed ([quickstart guide](https://docs.anthropic.com/en/docs/claude-code/overview))
- Repository cloned and dependencies installed: `uv sync`
- Verify Claude Code works: run `claude` from the repo root
- Familiarity with the UCI Wine dataset: 178 samples, 13 features (alcohol, malic acid, ash, etc.), 3 wine classes. Available via `sklearn.datasets.load_wine()`

## How Claude Code Features Work Together

### CLAUDE.md -- The Rulebook

`CLAUDE.md` files at the repo root and in `part1_claude_code/` define coding standards that Claude reads at session start and follows for every file it writes. These include rules for logging format, type annotations, function structure, library choices (polars not pandas), and more. Read both files before starting the lab.

### Hooks -- Automated Quality Gates

Hooks are configured in `.claude/settings.json` and fire automatically -- you do not invoke them. This project has three:

1. **PostToolUse command hook** (Write|Edit): Runs `scripts/check_python.sh` which executes `ruff check --fix`, `ruff format`, and `python -m py_compile` on every Python file Claude writes or edits.
2. **PostToolUse prompt hook** (Write|Edit): An LLM evaluates whether a test file should be created for each new Python file.
3. **PreToolUse command hook** (Bash): Runs `scripts/block_force_push.sh` to block any `git push --force` command.

Reference: [Claude Code Hooks](https://docs.anthropic.com/en/docs/claude-code/hooks)

### Slash Commands -- Structured Planning

The `/plan` command (defined in `.claude/commands/plan.md`) creates a detailed implementation plan in `.scratchpad/` and waits for your approval before building. This gives you a chance to review, modify, and sign off on the approach.

Reference: [Claude Code Slash Commands](https://docs.anthropic.com/en/docs/claude-code/slash-commands)

### Skills -- Reusable Workflows

Skills are markdown files in `.claude/skills/<name>/SKILL.md` that teach Claude reusable workflows. This project includes three:

- `/analyze-data`: Performs exploratory data analysis (statistics, distributions, correlations, outliers)
- `/evaluate-model`: Evaluates a trained model and generates a performance report
- `/generate-report`: Creates a comprehensive markdown report from model artifacts

Reference: [Claude Code Skills](https://docs.anthropic.com/en/docs/claude-code/skills)

### Subagents -- Task Decomposition

Claude spawns specialized subagents via the Task tool for focused subtasks. You will see these in action when Claude explores the codebase (Explore agent), designs the implementation (Plan agent), and runs commands (Bash agent).

Reference: [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)

## The Lab Task

### Staying Current

This space moves fast. Claude Code ships new features regularly, and the best practices evolve with them. To stay up to date:

- Follow [Boris Cherny](https://x.com/bcherny) (creator of Claude Code) and the [Anthropic Engineering blog](https://www.anthropic.com/engineering) for the latest tips, workflows, and feature announcements
- Read this thread on how he runs 5 Claudes in parallel with Opus 4.5: [https://x.com/bcherny/status/2007179832300581177](https://x.com/bcherny/status/2007179832300581177)
- See this post for additional workflow ideas: [https://x.com/i/status/2017742741636321619](https://x.com/i/status/2017742741636321619)

As you work through this lab, think about how you could combine what you learn here (plans, hooks, skills, subagents) with techniques from those posts to build your own workflow.

### Your Mission

Build a complete ML pipeline for classifying wines into 3 classes using the UCI Wine dataset (`sklearn.datasets.load_wine()`). The pipeline must include:

1. **Exploratory data analysis** with summary statistics, distribution plots, a correlation heatmap, class balance check, and outlier detection
2. **Feature engineering** with at least 3 derived features, standard scaling, and a stratified train/test split
3. **XGBoost classification model training** with 5-fold cross-validation and evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix)
4. **A comprehensive evaluation report** with metrics, feature importance, and recommendations

You will **not** write any Python code yourself. You will use Claude Code to plan and build the entire pipeline.

## Step-by-Step Walkthrough

### Step 1: Explore the Configuration

Before you start building, understand what Claude Code already knows about this project. Read the following files:

```bash
# Project rules Claude will follow
cat part1_claude_code/CLAUDE.md
cat CLAUDE.md

# Hook configuration
cat .claude/settings.json

# Hook scripts
cat scripts/check_python.sh
cat scripts/block_force_push.sh

# Skills
cat .claude/skills/analyze-data/SKILL.md
cat .claude/skills/evaluate-model/SKILL.md
cat .claude/skills/generate-report/SKILL.md

# Plan slash command
cat .claude/commands/plan.md
```

No Claude Code interaction yet -- this is manual reading to understand the setup.

### Step 2: Start Claude Code and Create a Plan

Open Claude Code from the repo root and use the `/plan` slash command. Write your own prompt based on the mission above -- be specific about what you want. Here is a minimal example to get you started:

```
/plan Build a Wine classification pipeline with EDA, feature engineering,
XGBoost with cross-validation, and an evaluation report. Use load_wine from
sklearn. Put scripts in part1_claude_code/src/ and output in output/.
```

The more detail you provide, the better the plan will be. Try adding specifics like the metrics you want, the number of CV folds, or the derived features you have in mind.

**What to Watch For:**
- Claude uses a **subagent** (Task tool with Explore type) to investigate the existing codebase before planning
- The plan is written to `.scratchpad/<feature-name>/plan.md`
- Claude stops and asks you to **review before building** -- it will not start coding yet
- The plan references CLAUDE.md coding standards in its technical decisions

### Step 3: Review and Refine the Plan

Read the plan Claude created. Then provide at least one modification. Some suggestions:

- "Add hyperparameter tuning using RandomizedSearchCV with at least 20 iterations"
- "Add a per-class precision/recall breakdown and a confusion matrix heatmap"
- "Include a --debug CLI flag that sets logging to DEBUG level"

Example prompt after reviewing:

```
I reviewed the plan. Please update it to also include hyperparameter tuning
using RandomizedSearchCV with 20 iterations and 5-fold stratified CV. The
tuning results should be saved to output/tuning_results.json. Then proceed
with implementation.
```

**What to Watch For:**
- Claude updates the plan file in `.scratchpad/`
- After you approve, Claude begins writing code

### Step 4: Watch Claude Build

Sit back and observe. Claude will create multiple Python files. Do not interrupt unless something looks clearly wrong.

**What to Watch For:**

| What Happens | Claude Code Feature | How You Can Tell |
|---|---|---|
| Claude writes a `.py` file | Write tool | You see the file content appear |
| Ruff auto-formats the file | PostToolUse command hook | You see "Running ruff and py_compile..." status message |
| py_compile verifies syntax | PostToolUse command hook | Same status message; no error means success |
| Claude checks if tests exist | PostToolUse prompt hook | Claude considers creating a test file after each write |
| Code uses `polars` not `pandas` | CLAUDE.md enforcement | Check the import statements |
| Logging uses the prescribed format | CLAUDE.md enforcement | Check the `logging.basicConfig()` call in each file |
| Private functions start with `_` | CLAUDE.md enforcement | Check function names in the code |
| Constants at top of file | CLAUDE.md enforcement | Look at the top of each file, not inside functions |
| Claude decomposes into subtasks | Subagents (Task tool) | You may see task spawning in the output |

### Step 5: Run the Pipeline

After Claude finishes building, run the scripts in order:

```bash
uv run python part1_claude_code/src/01_eda.py
uv run python part1_claude_code/src/02_feature_engineering.py
uv run python part1_claude_code/src/03_xgboost_model.py
```

Verify that the `output/` directory contains distribution plots, correlation matrix, parquet files, model file, confusion matrix, feature importance chart, and evaluation report.

**What to Watch For:**
- The logging output uses the exact format from CLAUDE.md
- Elapsed time is reported at the end of each script

### Step 6: Use the Skills

Now use the pre-built skills to analyze and report on what Claude built:

```
/analyze-data
```

Then:

```
/evaluate-model
```

Then:

```
/generate-report output/
```

**What to Watch For:**
- Each skill loads its instructions from `.claude/skills/<name>/SKILL.md`
- Claude follows the skill steps (compare against the SKILL.md you read in Step 1)
- The `/generate-report` skill uses the template from `templates/report_template.md`
- Output is saved to `output/`

### Step 7: Trigger a Hook Intentionally

Ask Claude to do something that will trigger a hook, so you can observe it directly:

```
Create a new Python file part1_claude_code/src/utils.py with a helper function
that prints "hello world" without using logging and without type annotations.
```

**What to Watch For:**
- The PostToolUse hook runs ruff and py_compile immediately after the file is written
- The prompt hook may flag the missing test file
- If Claude follows CLAUDE.md, it will likely add logging and type annotations despite your instruction -- this shows CLAUDE.md influence
- If Claude does write non-compliant code, the hooks catch it

### Step 8: Test the Safety Hook

Ask Claude to force push:

```
Run git push --force origin main
```

**What to Watch For:**
- The PreToolUse hook blocks the command **before** it executes
- Claude reports that force push is blocked by project hooks

## Feature Summary Checklist

After completing the lab, confirm you observed each feature:

- [confirmed] **CLAUDE.md**: Code uses polars, not pandas
- [confirmed] **CLAUDE.md**: Logging format matches the prescribed pattern
- [confirmed] **CLAUDE.md**: Private functions start with underscore
- [confirmed] **CLAUDE.md**: Constants declared at file top, not hardcoded in functions
- [confirmed] **CLAUDE.md**: Functions are under 50 lines
- [confirmed] **Hook (command)**: Saw "Running ruff and py_compile..." status message
- [confirmed] **Hook (prompt)**: Claude created or considered creating test files
- [confirmed] **Hook (command)**: Force push was blocked
- [confirmed] **Slash command**: `/plan` created a plan in `.scratchpad/`
- [confirmed] **Slash command**: Plan followed the template from `.claude/commands/plan.md`
- [confirmed] **Skill**: `/analyze-data` produced EDA analysis
- [confirmed] **Skill**: `/evaluate-model` produced evaluation output
- [confirmed] **Skill**: `/generate-report` produced `full_report.md`
- [confirmed] **Subagent**: Claude explored existing code before implementing

## Reference Material

The `demo/` folder contains reference implementations and the original granular exercises:

- `demo/solved/` -- Pre-built pipeline scripts that show one possible correct implementation
- `demo/exercises/` -- The original three separate exercises, now consolidated into the single lab above

You can compare your Claude-generated code against `demo/solved/` to see how implementations differ.

| File | Description |
|------|-------------|
| `demo/solved/01_eda.py` | Loads California Housing, computes statistics with polars, generates distribution and correlation plots |
| `demo/solved/02_feature_engineering.py` | Creates derived features, handles infinite values, scales features, splits into train/test |
| `demo/solved/03_xgboost_model.py` | Trains XGBoost regressor with CV and hyperparameter tuning, computes metrics (RMSE, MAE, R2) |
| `demo/solved/04_generate_report.py` | Generates a comprehensive markdown report from model artifacts |

## Troubleshooting

- **Claude is not following CLAUDE.md rules**: Make sure you started Claude Code from the repo root, not from a subdirectory.
- **Hooks are not firing**: Check that `.claude/settings.json` exists at the repo root and has the correct structure. Run `cat .claude/settings.json` to verify.
- **Skills not found**: Skills must be in `.claude/skills/<name>/SKILL.md`. Verify with `ls .claude/skills/`.
- **`uv run` fails**: Run `uv sync` from the repo root first.
- **Output directory missing**: The scripts create `output/` automatically, but you can also run `mkdir -p output`.

## Further Reading

- [Claude Code Memory (CLAUDE.md)](https://docs.anthropic.com/en/docs/claude-code/memory)
- [Claude Code Hooks](https://docs.anthropic.com/en/docs/claude-code/hooks)
- [Claude Code Skills](https://docs.anthropic.com/en/docs/claude-code/skills)
- [Claude Code Best Practices (Subagents)](https://www.anthropic.com/engineering/claude-code-best-practices)
