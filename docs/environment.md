# Environment

This project is intended to run from a local Python virtual environment, not a container.

Tracked files:

- `pyproject.toml`: package metadata, runtime dependencies, extras, and CLI entry point.
- `requirements.txt`: broad install requirements for app, dev, and test work.
- `requirements-notebook.txt`: optional minimal IDE/Jupyter kernel dependency.
- `requirements-lock.txt`: exact package versions from the current local `.venv`.

Ignored files:

- `.venv/`: local virtual environment.
- `runs/`: generated example and CLI outputs.
- `artifacts/`: model or data artifacts.

## Recreate The Local Environment

```bash
uv venv --python /opt/homebrew/bin/python3.11 .venv
uv pip install -r requirements.txt
uv pip install -e .
```

For the same package versions captured from this environment:

```bash
uv pip install -r requirements-lock.txt
uv pip install -e .
```

For IDE notebook cells:

```bash
uv pip install -r requirements-notebook.txt
```

Registering the kernel is optional and writes outside the repo:

```bash
.venv/bin/python -m ipykernel install --user --name graph-to-vec \
  --display-name "Python (graph-to-vec)"
```

## Update The Lock Snapshot

After intentionally changing dependencies:

```bash
uv pip freeze --exclude-editable > requirements-lock.txt
```

Do not commit `.venv/` itself. Commit the manifest changes instead.
