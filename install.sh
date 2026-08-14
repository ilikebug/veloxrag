#!/usr/bin/env bash
# VeloxRAG installer.
#
#   curl -fsSL https://raw.githubusercontent.com/ilikebug/veloxrag/main/install.sh | bash
#
# Brings up a local RAG service that acts as memory for coding agents: installs
# Ollama if it is missing, pulls the embedding model, writes a compose file, and
# starts the stack. Re-running is safe; it converges rather than reinstalling,
# and never touches existing data volumes.
#
# Docker is the one prerequisite this script will not install for you. Installing
# a container runtime rewrites system state and, on macOS, needs a choice between
# Docker Desktop and Colima that belongs to you.
set -euo pipefail

VELOX_REF="${VELOX_REF:-main}"
VELOX_HOME="${VELOX_HOME:-$HOME/.veloxrag}"
VELOX_MODEL="${VELOX_MODEL:-bge-m3}"
COMPOSE_URL="https://raw.githubusercontent.com/ilikebug/veloxrag/${VELOX_REF}/compose.yaml"
VELOXRAG_URL="https://raw.githubusercontent.com/ilikebug/veloxrag/${VELOX_REF}/veloxrag"
API_PORT="${RAG_API_HOST_PORT:-8000}"
# Only used when Colima is installed and not yet running; see below.
VELOX_VM_CPU="${VELOX_VM_CPU:-4}"
VELOX_VM_MEMORY="${VELOX_VM_MEMORY:-8}"
VELOX_VM_DISK="${VELOX_VM_DISK:-60}"

say() { printf '\033[1m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$1" >&2; }
die() { printf '\033[31merror:\033[0m %s\n' "$1" >&2; exit 1; }

# --------------------------------------------------------------------------
# Prerequisites
# --------------------------------------------------------------------------

case "$(uname -s)" in
  Darwin) OS=macos ;;
  Linux) OS=linux ;;
  *) die "unsupported operating system: $(uname -s). This installs on macOS and Linux." ;;
esac

command -v curl >/dev/null 2>&1 || die "curl is required"

if ! command -v docker >/dev/null 2>&1; then
  if [ "$OS" = macos ]; then
    die "Docker is required. Install Docker Desktop (https://docs.docker.com/desktop/) or Colima (brew install colima docker && colima start), then run this again."
  fi
  die "Docker is required. See https://docs.docker.com/engine/install/ then run this again."
fi

# Started only when nothing is running yet. Resizing a live VM is a different
# matter: `colima start` does not change a running instance's resources, but
# --save-config defaults to true, so passing --cpu/--memory to a running Colima
# rewrites the config without applying it and silently changes what the machine
# does at its next restart. That is the same class of fault as a setting the
# service reads but compose never passes — config and reality disagreeing, with
# nothing reporting it. So a running VM is left exactly as the user sized it.
if ! docker info >/dev/null 2>&1; then
  if command -v colima >/dev/null 2>&1 && ! colima status >/dev/null 2>&1; then
    say "Starting Colima with ${VELOX_VM_CPU} CPUs, ${VELOX_VM_MEMORY} GiB of memory and ${VELOX_VM_DISK} GiB of disk"
    say "Override with VELOX_VM_CPU / VELOX_VM_MEMORY / VELOX_VM_DISK; a Colima disk can grow later but not shrink"
    colima start --cpu "$VELOX_VM_CPU" --memory "$VELOX_VM_MEMORY" --disk "$VELOX_VM_DISK" \
      || die "colima start failed. Size it yourself and run this again."
  fi
fi

if ! docker info >/dev/null 2>&1; then
  die "Docker is installed but not running. Start Docker Desktop, or run 'colima start', then run this again."
fi

# Inline compose configs are how the embedding proxy carries its own nginx
# configuration, so a single downloaded file is the whole stack. They landed in
# 2.23.1, and an older Compose parses the file but mounts nothing.
compose_version="$(docker compose version --short 2>/dev/null || echo 0)"
if [ "$(printf '%s\n2.23.1\n' "$compose_version" | sort -V | head -1)" != "2.23.1" ]; then
  die "Docker Compose 2.23.1 or newer is required (found ${compose_version}); this file uses inline configs."
fi

# Warned about rather than fixed. Resizing the VM is a system-level decision, and
# on macOS the runtime might be Docker Desktop, Colima, Rancher or OrbStack — and
# for Colima specifically `colima start --cpu N` does not resize a running
# instance, so "helpfully" applying it would mean stopping a VM that may be
# running someone else's containers.
docker_cpus="$(docker info --format '{{.NCPU}}' 2>/dev/null || echo 0)"
docker_mem_gib="$(docker info --format '{{.MemTotal}}' 2>/dev/null | awk '{printf "%d", $1/1073741824}')"
if [ "${docker_cpus:-0}" -lt 2 ] 2>/dev/null; then
  warn "Docker reports ${docker_cpus} CPU(s). Ingestion chunks on one core and will be slow."
fi
if [ "${docker_mem_gib:-0}" -lt 4 ] 2>/dev/null; then
  warn "Docker reports ${docker_mem_gib} GiB of memory. The containers idle near 450 MiB, but 4 GiB is the comfortable floor."
fi
if [ "${docker_cpus:-0}" -lt 2 ] 2>/dev/null || [ "${docker_mem_gib:-0}" -lt 4 ] 2>/dev/null; then
  if command -v colima >/dev/null 2>&1; then
    warn "To resize Colima: colima stop && colima start --cpu 4 --memory 8 --disk 60"
  else
    warn "Docker Desktop sets these under Settings -> Resources."
  fi
fi

# Disk is the one that fails worst and the one this cannot read: the daemon does
# not report free space, and measuring it means running a container. A full disk
# makes Qdrant refuse writes with "No space left on device" and makes a Docker
# build fail without saying why, so it is worth checking by hand before a large
# ingest: docker system df, and df inside the VM.
say "About 4 GB of disk is needed for the images, the model and a small corpus"

# --------------------------------------------------------------------------
# Ollama: the embedding model runs on the host, because that is the only place
# it reaches the GPU. Docker on macOS is a Linux VM with no Metal passthrough,
# measured at 3.1 chunks/s against 14.2 on the host.
# --------------------------------------------------------------------------

if command -v ollama >/dev/null 2>&1; then
  say "Ollama already installed ($(ollama --version 2>/dev/null | head -1 || echo present))"
else
  say "Installing Ollama"
  if [ "$OS" = macos ] && command -v brew >/dev/null 2>&1; then
    brew install ollama
  else
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  command -v ollama >/dev/null 2>&1 || die "Ollama installation did not put 'ollama' on PATH"
fi

# Started as a managed service rather than a bare background process, so that it
# survives the shell that launched it — but deliberately not registered to launch
# at login. Nothing here adopts a startup item on the user's behalf.
start_ollama() {
  if curl -fsS --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    return 0
  fi
  # `run` rather than `start`, and `start` rather than `enable --now`: neither
  # registers anything to launch at login. Starting the machine is the user's
  # decision, and `veloxrag start` is how the stack comes back afterwards.
  if [ "$OS" = macos ] && command -v brew >/dev/null 2>&1 && brew list ollama >/dev/null 2>&1; then
    brew services run ollama >/dev/null 2>&1 || true
  elif [ "$OS" = linux ] && command -v systemctl >/dev/null 2>&1; then
    sudo systemctl start ollama >/dev/null 2>&1 || true
  fi
  for _ in $(seq 1 30); do
    curl -fsS --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

say "Starting Ollama"
if ! start_ollama; then
  warn "Could not reach Ollama on 127.0.0.1:11434 as a managed service; falling back to a background process."
  nohup ollama serve >"${TMPDIR:-/tmp}/veloxrag-ollama.log" 2>&1 &
  for _ in $(seq 1 30); do
    curl -fsS --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
    sleep 1
  done
  curl -fsS --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1 \
    || die "Ollama did not become reachable on 127.0.0.1:11434"
fi

if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "${VELOX_MODEL}:latest\|${VELOX_MODEL}"; then
  say "Embedding model ${VELOX_MODEL} already present"
else
  say "Pulling embedding model ${VELOX_MODEL} (about 1.2 GB, once)"
  ollama pull "${VELOX_MODEL}"
fi

# --------------------------------------------------------------------------
# The stack
# --------------------------------------------------------------------------

mkdir -p "$VELOX_HOME"
say "Fetching compose.yaml into ${VELOX_HOME}"
curl -fsSL "$COMPOSE_URL" -o "${VELOX_HOME}/compose.yaml.new"
[ -s "${VELOX_HOME}/compose.yaml.new" ] || die "downloaded compose.yaml is empty; check ${COMPOSE_URL}"
mv "${VELOX_HOME}/compose.yaml.new" "${VELOX_HOME}/compose.yaml"

# --------------------------------------------------------------------------
# The veloxrag command. A reboot does not restore a working stack on its own:
# the containers carry restart: unless-stopped, so the daemon brings some of them
# back, but a daemon-driven restart does not honour depends_on — that applies
# only to `docker compose up`. Measured after a VM stop/start, three of seven
# containers returned and the worker sat in a restart loop. This command is what
# repairs that, and it is a command rather than a login item on purpose.
# --------------------------------------------------------------------------

install_command() {
  local dest="" candidate
  # A directory already on PATH is worth more than a tidier one that is not,
  # because a command the user cannot invoke is not installed.
  for candidate in "$HOME/.local/bin" "$(command -v brew >/dev/null 2>&1 && brew --prefix 2>/dev/null || echo /usr/local)/bin" /usr/local/bin; do
    case ":${PATH}:" in *":${candidate}:"*) ;; *) continue ;; esac
    if [ -d "$candidate" ] && [ -w "$candidate" ]; then dest="$candidate"; break; fi
  done
  # Nothing writable on PATH: fall back to a conventional location and say what
  # it costs, rather than failing the whole install over a convenience command.
  if [ -z "$dest" ]; then
    dest="$HOME/.local/bin"
    mkdir -p "$dest" || { warn "could not create ${dest}; skipping the veloxrag command"; return 1; }
    warn "${dest} is not on your PATH. Add it to use \`veloxrag\`:"
    warn "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc"
  fi
  curl -fsSL "$VELOXRAG_URL" -o "${dest}/veloxrag.new" 2>/dev/null || {
    warn "could not download the veloxrag command from ${VELOXRAG_URL}"; return 1; }
  [ -s "${dest}/veloxrag.new" ] || { warn "downloaded veloxrag is empty"; rm -f "${dest}/veloxrag.new"; return 1; }
  chmod +x "${dest}/veloxrag.new"
  mv "${dest}/veloxrag.new" "${dest}/veloxrag"
  VELOX_COMMAND_PATH="${dest}/veloxrag"
  say "Installed the veloxrag command at ${VELOX_COMMAND_PATH}"
}

VELOX_COMMAND_PATH=""
install_command || true

say "Starting the stack (first run pulls images and initializes the knowledge base)"
cd "$VELOX_HOME"
COMPOSE_DISABLE_ENV_FILE=1 docker compose up -d

say "Waiting for initialization to finish"
COMPOSE_DISABLE_ENV_FILE=1 docker compose wait bootstrap >/dev/null 2>&1 || true

ready=0
for _ in $(seq 1 60); do
  if curl -fsS --max-time 3 "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[ "$ready" = 1 ] || die "the API did not answer on 127.0.0.1:${API_PORT}. Inspect with: cd ${VELOX_HOME} && docker compose logs api"

# --------------------------------------------------------------------------
# What to do next
# --------------------------------------------------------------------------

cat <<EOF

$(printf '\033[32m✓\033[0m') VeloxRAG is running at http://127.0.0.1:${API_PORT}

Connect an agent over MCP:

    claude mcp add --scope user rag-memory -- uvx --from git+https://github.com/ilikebug/veloxrag velox-mcp

Or, for a client that takes a config file:

    {
      "mcpServers": {
        "rag-memory": {
          "command": "uvx",
          "args": ["--from", "git+https://github.com/ilikebug/veloxrag", "velox-mcp"]
        }
      }
    }

It exposes four read-only tools: search_memory, read_document, list_documents and
memory_status. No token and no knowledge base id are needed.

Relevance judgement is left to the agent: retrieve more passages than you need,
then widen the promising ones with read_document before deciding. The answer
often sits just outside the passage that matched.

Everyday commands:

    veloxrag status              what is running, and whether Ollama and the API answer
    veloxrag log [service]       follow logs; all services, or one such as api or worker
    veloxrag restart             restart the containers
    veloxrag stop                stop the containers, leaving Ollama and Docker alone

After a reboot, one command brings everything back:

    veloxrag start

    It starts Docker, then Ollama, then the containers, in that order. Nothing is
    registered to launch at login: that choice is yours. The reason this is not
    automatic is that it cannot be — the containers are marked
    restart: unless-stopped, but a daemon-driven restart does not honour
    depends_on, so after a reboot some come back and the worker can land in a
    restart loop. \`veloxrag start\` is what repairs the order.
$(if [ -n "$VELOX_COMMAND_PATH" ]; then
    echo "    Installed at ${VELOX_COMMAND_PATH}."
  else
    echo "    The command could not be installed; see the warning above. Until then:"
    echo "    cd ${VELOX_HOME} && docker compose up -d"
  fi)

To erase the knowledge base and start over:

    cd ${VELOX_HOME} && docker compose down -v

EOF
