# Agent Instructions

## Project Context

**BasketTube** is an aegean.ai tech demonstrator for AI-powered basketball game analysis. It combines:

- **Computer vision** — player detection (RF-DETR), segmentation/tracking (SAM2), team classification, jersey OCR, court mapping
- **Speech-to-text** — commentary transcription and semantic parsing
- **Vision-language models** — SmolVLM2, SigLIP for action recognition and play classification

The project has two main surfaces:
1. **Jupyter notebooks** (`notebooks/`) — CV pipeline prototyping, runs in a GPU Docker container
2. **Web application** — FastAPI backend (`api/`) + Next.js/shadcn frontend (`frontend/`) for the full interactive experience

## Development Workflow

### Notebooks (GPU)

```bash
docker compose --profile nvidia up -d
# JupyterLab at http://localhost:8888
```

Source notebooks live in `notebooks/`, mounted into the container at `/workspace/notebooks`.

### Backend / Frontend

The FastAPI API (`api/src/`) and Next.js frontend (`frontend/`) are retained from the inherited codebase and will be adapted for BasketTube's needs.

## Key Commands

```bash
# Build and start notebook container
docker compose --profile nvidia build
docker compose --profile nvidia up -d

# View logs
docker compose --profile nvidia logs -f notebook

# Stop
docker compose --profile nvidia down
```

## Environment Setup

Copy `.env.example` to `.env` and set:
- `HF_TOKEN` — Hugging Face API token
- `ROBOFLOW_API_KEY` — Roboflow API key

## Dependencies

Managed via `pyproject.toml` with `uv`. To add a package:
1. Add it to `pyproject.toml`
2. Run `uv lock`
3. Rebuild the Docker image

<!-- BEGIN BEADS INTEGRATION -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Dolt-powered version control with native sync
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

**Create new issues:**

```bash
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd update <id> --claim --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task atomically**: `bd update <id> --claim`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Auto-Sync

bd automatically syncs via Dolt:

- Each write auto-commits to Dolt history
- Use `bd dolt push`/`bd dolt pull` for remote sync
- No manual export/import needed!

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and docs/QUICKSTART.md.

<!-- END BEADS INTEGRATION -->

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
