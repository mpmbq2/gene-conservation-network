# gene-conservation-network Agent Guidelines

This file contains guidelines for automated agents (and human developers) working on the `gene-conservation-network` repository.
It covers environment setup, build/test commands, and code style conventions.

## 1. Environment & Dependencies

This project uses **Pixi** for dependency management and environment handling, and **DVC** for data pipelines.

-   **Lock File**: `pixi.lock` (Do not edit manually).
-   **Configuration**: `pyproject.toml` (See `[tool.pixi]`).

### Commands

-   **Install Dependencies**:
    ```bash
    pixi install
    ```

-   **Activate Shell**:
    ```bash
    pixi shell
    ```
    *Note: Ensure you are in the pixi shell or using the configured python interpreter before running other commands.*

## 2. Build, Lint, and Test

We use `ruff` for linting and formatting, and `pytest` for testing. Tasks are defined in `pyproject.toml`.

### Linting & Formatting

-   **Check Code Quality (Lint)**:
    ```bash
    pixi run lint
    # Runs: ruff format --check && ruff check
    ```

-   **Fix Code Quality (Format)**:
    ```bash
    pixi run format
    # Runs: ruff check --fix && ruff format
    ```

-   **Configuration**: Rules are defined in `pyproject.toml` under `[tool.ruff]`.
    -   Line length: 99 characters.
    -   Import sorting: configured to treat `gene_conservation_network` as first-party.

### Testing

-   **Run All Tests**:
    ```bash
    pixi run test
    # Runs: pytest tests
    ```

-   **Run a Single Test File**:
    ```bash
    pixi run pytest tests/test_data.py
    ```

-   **Run a Single Test Case**:
    ```bash
    pixi run pytest tests/test_data.py::test_code_is_tested
    ```

-   **Run with Output (s)**:
    ```bash
    pixi run pytest -s tests/test_data.py
    ```

### Pipelines (DVC)

-   Data pipelines are managed with **DVC**.
-   See `dvc.yaml` for pipeline stages.
-   Execute pipelines using `dvc repro` or run specific stages as defined in your workflow.

## 3. Code Style & Conventions

### General Python

-   **Version**: Python 3.11+ (specifically `~=3.11.0`).
-   **Naming**:
    -   Variables/Functions: `snake_case`
    -   Classes: `PascalCase`
    -   Constants: `UPPER_CASE`
-   **Type Hints**: Use standard Python type hints for function arguments and return values.
    ```python
    def process_data(input_path: Path, verbose: bool = False) -> dict:
        ...
    ```

### Imports

-   **Sorting**: Imports must be sorted. Use `pixi run format` to apply `isort` rules via `ruff`.
-   **Structure**:
    1.  Standard library imports (e.g., `pathlib`, `sys`)
    2.  Third-party imports (e.g., `loguru`, `tqdm`, `typer`)
    3.  First-party imports (e.g., `gene_conservation_network.config`)
-   **Style**: Prefer absolute imports over relative imports for clarity.

### Logging & Output

-   **Library**: Use `loguru` for all logging.
-   **Rule**: **Do not use `print()`** for application logging. Use `logger.info()`, `logger.debug()`, `logger.error()`, etc.
    ```python
    from loguru import logger

    logger.info("Starting processing...")
    logger.success("Processing complete.")  # Use success for completion messages
    logger.error("An error occurred: {}", error_msg)  # Prefer {} placeholders
    ```
-   **Progress Bars**: Use `tqdm` for loops that take time.
    ```python
    from tqdm import tqdm
    for item in tqdm(items, desc="Processing items"):
        process(item)
    ```
-   **Logger Configuration**: `loguru` is integrated with `tqdm` in `config.py` - no additional setup needed.

### CLI Applications

-   **Library**: Use `typer` for building command-line interfaces.
-   **Structure**:
    ```python
    import typer
    app = typer.Typer()

    @app.command()
    def main(name: str):
        ...

    if __name__ == "__main__":
        app()
    ```

### File System & Paths

-   **Library**: Use `pathlib.Path` instead of `os.path` strings.
-   **Configuration**: define common paths in `gene_conservation_network/config.py` (e.g., `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`).

### Error Handling

-   Use specific exception types where possible.
-   Let `loguru` handle exception formatting when logging errors.

## 4. Project Structure

-   `gene_conservation_network/`: Source code package.
-   `tests/`: Test files (mirrors source structure where possible).
-   `data/`: Data directory (managed by DVC, often ignored by git).
-   `scripts/`: Standalone scripts (pipeline components).
-   `pyproject.toml`: Configuration for build, tools, and dependencies.
-   `dvc.yaml`: DVC pipeline definitions.

## 5. Agent Workflow Tips

1.  **Always Run Tests**: Before finishing a task, verify changes with `pixi run test`. If adding a new feature, add a corresponding test.
2.  **Lint Frequently**: Run `pixi run format` to automatically fix simple style issues and `pixi run lint` to catch other errors.
3.  **Check Config**: Look at `pyproject.toml` if unsure about tool settings.
4.  **Use Context**: Read surrounding code to match existing patterns, especially for logging and typing.

## 6. Copilot/Cursor Rules (Specifics)

*No specific `.cursorrules` or `.github/copilot-instructions.md` found in this repository at the time of writing.*
*Adhere to the conventions above as the primary source of truth.*

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
