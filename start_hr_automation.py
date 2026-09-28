from __future__ import annotations

import shutil
import socket
import subprocess
import time
import webbrowser
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"C:\Projects\HR-Automation")
PROJECT_GIT_PATH = "/c/Projects/HR-Automation"

PYTHON_SERVICE_GIT_PATH = f"{PROJECT_GIT_PATH}/python-service"

N8N_CONTAINER = "hr-automation-n8n"

N8N_PORT = 5678
API_PORT = 8000

WORKFLOW_URL = (
    "http://localhost:5678/workflow/h8tKqYv48gW1nKgQ"
)

DOCKER_DESKTOP = Path(
    r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
)

# Maximum time to wait for services to become available.
SERVICE_TIMEOUT = 90


# ============================================================
# HELPERS
# ============================================================

def print_step(message: str) -> None:
    print(f"\n{'=' * 70}")
    print(message)
    print(f"{'=' * 70}")


def find_git_installation() -> tuple[Path, Path]:
    """
    Returns:
        (git-bash.exe, bash.exe)
    """

    possible_roots = [
        Path(r"C:\Program Files\Git"),
        Path(r"C:\Program Files (x86)\Git"),
    ]

    # Try standard installation paths first.
    for root in possible_roots:
        git_bash = root / "git-bash.exe"
        bash = root / "bin" / "bash.exe"

        if git_bash.exists() and bash.exists():
            return git_bash, bash

    # Fallback: try PATH.
    bash_from_path = shutil.which("bash.exe")

    if bash_from_path:
        bash_path = Path(bash_from_path)

        # Usually ...\Git\bin\bash.exe
        git_root = bash_path.parent.parent
        git_bash = git_root / "git-bash.exe"

        if git_bash.exists():
            return git_bash, bash_path

    raise FileNotFoundError(
        "Git Bash was not found.\n"
        "Expected installation such as:\n"
        r"C:\Program Files\Git\git-bash.exe"
    )


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    """Return True when something is listening on the given port."""

    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def wait_for_port(
    port: int,
    timeout: int = SERVICE_TIMEOUT,
    label: str = "service",
) -> bool:
    """Wait until a TCP port becomes available."""

    print(f"Waiting for {label} on port {port}...")

    start = time.time()

    while time.time() - start < timeout:
        if port_is_open(port):
            print(f"[OK] {label} is available on port {port}.")
            return True

        time.sleep(2)

    print(f"[ERROR] {label} did not become available within {timeout} seconds.")
    return False


def docker_is_available() -> bool:
    """Check whether Docker Engine is responding."""

    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )

        return result.returncode == 0

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def start_docker_desktop() -> None:
    """Start Docker Desktop if it isn't already running."""

    print_step("STEP 1: Checking Docker")

    if docker_is_available():
        print("[OK] Docker Engine is already running.")
        return

    if not DOCKER_DESKTOP.exists():
        raise FileNotFoundError(
            f"Docker Desktop was not found at:\n{DOCKER_DESKTOP}"
        )

    print("Docker Engine is not responding.")
    print("Starting Docker Desktop...")

    subprocess.Popen(
        [str(DOCKER_DESKTOP)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("Waiting for Docker Engine...")

    start = time.time()

    while time.time() - start < SERVICE_TIMEOUT:
        if docker_is_available():
            print("[OK] Docker Engine is ready.")
            return

        time.sleep(3)

    raise RuntimeError(
        "Docker Desktop started, but Docker Engine did not become ready."
    )


def open_git_bash(
    bash_path: Path,
    command: str,
    title: str,
) -> subprocess.Popen:
    """
    Open a real Git Bash / Mintty window and execute the command.
    """

    git_root = bash_path.parent.parent
    mintty = git_root / "usr" / "bin" / "mintty.exe"

    if not mintty.exists():
        raise FileNotFoundError(
            f"Git Bash terminal (mintty) was not found at:\n{mintty}"
        )

    process = subprocess.Popen(
        [
            str(mintty),
            "--title",
            title,
            "-e",
            str(bash_path),
            "--login",
            "-i",
            "-c",
            command,
        ],
        cwd=str(PROJECT_DIR),
    )

    return process


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\nHR AUTOMATION STARTUP")
    print("=====================")
    print(f"Project : {PROJECT_DIR}")
    print(f"Workflow: {WORKFLOW_URL}")

    # --------------------------------------------------------
    # Validate project
    # --------------------------------------------------------

    if not PROJECT_DIR.exists():
        raise FileNotFoundError(
            f"Project directory not found:\n{PROJECT_DIR}"
        )

    # --------------------------------------------------------
    # Find Git Bash
    # --------------------------------------------------------

    print_step("STEP 2: Finding Git Bash")

    git_bash_exe, bash_exe = find_git_installation()

    print(f"[OK] Git Bash : {git_bash_exe}")
    print(f"[OK] Bash     : {bash_exe}")

    # --------------------------------------------------------
    # Start Docker Desktop if needed
    # --------------------------------------------------------

    start_docker_desktop()

    # --------------------------------------------------------
    # Open Git Bash #1
    # Start / verify n8n
    # --------------------------------------------------------

    print_step("STEP 3: Starting n8n")

    n8n_command = f"""
cd {PROJECT_GIT_PATH} || exit 1

echo ""
echo "=============================================="
echo " HR AUTOMATION - n8n"
echo "=============================================="
echo ""

echo "[1/3] Running docker ps..."
docker ps

echo ""
echo "[2/3] Checking n8n container..."

if docker ps --format '{{{{.Names}}}}' | grep -Fxq "{N8N_CONTAINER}"; then
    echo "[OK] {N8N_CONTAINER} is already running."
else
    echo "[INFO] n8n is not running."
    echo "[INFO] Starting n8n with Docker Compose..."
    echo ""

    docker compose -f n8n/docker-compose.yml up -d

    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Failed to start n8n."
        echo ""
        exec bash
    fi
fi

echo ""
echo "[3/3] Verifying docker ps..."
echo ""

docker ps

echo ""
echo "n8n startup command completed."
echo "This terminal will remain open."
echo ""

exec bash
"""

    n8n_terminal = open_git_bash(
        bash_exe,
        n8n_command,
        "HR Automation - n8n",
    )

    print("[OK] n8n Git Bash terminal opened.")

    # --------------------------------------------------------
    # Wait for n8n UI
    # --------------------------------------------------------

    if not wait_for_port(
        N8N_PORT,
        SERVICE_TIMEOUT,
        "n8n",
    ):
        print(
            "\n[ERROR] n8n UI is not available.\n"
            "Check the n8n Git Bash window for errors."
        )
        return

    # --------------------------------------------------------
    # Open Git Bash #2
    # Start FastAPI
    # --------------------------------------------------------

    print_step("STEP 4: Starting FastAPI")

    api_command = f"""
cd {PYTHON_SERVICE_GIT_PATH} || exit 1

echo ""
echo "=============================================="
echo " HR AUTOMATION - FASTAPI"
echo "=============================================="
echo ""

echo "[1/3] Project directory:"
pwd

echo ""
echo "[2/3] Activating virtual environment..."

if [ ! -f ".venv/Scripts/activate" ]; then
    echo "[ERROR] Python virtual environment not found."
    echo "Expected: .venv/Scripts/activate"
    echo ""
    exec bash
fi

source .venv/Scripts/activate

echo ""
echo "[OK] Virtual environment activated."
echo "Python:"
python --version

echo ""
echo "[3/3] Starting FastAPI..."
echo ""
echo "API URL:"
echo "http://localhost:8000"
echo ""
echo "Swagger docs:"
echo "http://localhost:8000/docs"
echo ""

uvicorn app:app --host 0.0.0.0 --port 8000

echo ""
echo "FastAPI process stopped."
echo ""

exec bash
"""

    api_terminal = open_git_bash(
        bash_exe,
        api_command,
        "HR Automation - FastAPI",
    )

    print("[OK] FastAPI Git Bash terminal opened.")

    # --------------------------------------------------------
    # Wait for FastAPI
    # --------------------------------------------------------

    api_ready = wait_for_port(
        API_PORT,
        SERVICE_TIMEOUT,
        "FastAPI",
    )

    if not api_ready:
        print(
            "\n[ERROR] FastAPI is not available on port 8000.\n"
            "Check the FastAPI Git Bash window for the actual error."
        )
        return

    # --------------------------------------------------------
    # Open n8n workflow
    # --------------------------------------------------------

    print_step("STEP 5: Opening n8n workflow")

    print("[OK] n8n is running.")
    print("[OK] FastAPI is running.")
    print()
    print(f"Opening:")
    print(WORKFLOW_URL)

    time.sleep(2)

    webbrowser.open(WORKFLOW_URL)

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print_step("HR AUTOMATION IS READY")

    print("n8n:")
    print("  http://localhost:5678")

    print()
    print("FastAPI:")
    print("  http://localhost:8000")

    print()
    print("Swagger:")
    print("  http://localhost:8000/docs")

    print()
    print("Workflow:")
    print(f"  {WORKFLOW_URL}")

    print()
    print("Both Git Bash terminals have been left open.")
    print()
    print("Your current n8n trigger is manual, so after the browser")
    print("opens you still need to click 'Execute workflow'.")

    # Keep this Python terminal alive briefly so the user can
    # read the status if launched by double-click.
    time.sleep(8)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nStartup cancelled by user.")

    except Exception as exc:
        print("\n[ERROR]")
        print(str(exc))
        print()
        input("Press Enter to close...")