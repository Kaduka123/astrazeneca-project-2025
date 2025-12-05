# CHO Analysis - Development Setup

This repository contains tools and analysis for gene expression in CHO cell lines for marker identification and sequence feature characterization.

## Development Environment Setup

### Prerequisites

Before you begin, install the following software:

- **VS Code**: [Download Visual Studio Code](https://code.visualstudio.com/download)
- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **VS Code Extensions**:
  - Remote - Containers ([ms-vscode-remote.remote-containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers))

### Getting Started

1. **Clone the repository**

```bash
git clone <repository-url>
cd cho-analysis
```

2. **Start Docker** (if not already running)

- Windows/macOS: Docker Desktop should be running
- Linux/NixOS: `sudo systemctl start docker`

3. **Open in VS Code**

```bash
code .
```

4. **Launch the Development Container**

- When VS Code opens, you'll see a notification that a dev container configuration was detected
- Click "Reopen in Container"
- Alternatively, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS), type "Remote-Containers: Reopen in Container" and select it

5. **Wait for container setup to complete**

- VS Code will build a Docker container with all required dependencies
- This may take a few minutes on first run

6. Execute the project

```bash
# For task 1
python -m scripts.run_analysis --task 1 --methods spearman pearson --top-n 20

# For task 2
python -m scripts.run_analysis --task 2
```

## Development Workflow

- Code is formatted using `ruff` and checked with `mypy`.
- Run tests with `pytest`.
- Jupyter notebooks can be used for exploration and visualization.

## Troubleshooting

If you encounter issues with the development container:

1. Verify Docker is running
2. Try rebuilding the container: `Ctrl+Shift+P` > "Remote-Containers: Rebuild Container"
3. Check the Docker logs: `docker logs <container-id>`

---

For additional information, see the project documentation.