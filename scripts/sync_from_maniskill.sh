#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_REPO="${SOURCE_REPO:-/home/hebu/code/ManiSkill}"

COMMIT_MSG=""
DO_PUSH=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  sync_from_maniskill.sh
  sync_from_maniskill.sh -m "your commit message"
  sync_from_maniskill.sh -m "your commit message" --push
  sync_from_maniskill.sh --dry-run

Options:
  -m, --message   Commit message. If omitted, only sync files.
  --push          Push after commit. Must be used with -m/--message.
  --dry-run       Print planned rsync changes without writing files.

Environment:
  SOURCE_REPO     Override source ManiSkill path.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        exit 1
      fi
      COMMIT_MSG="$2"
      shift 2
      ;;
    --push)
      DO_PUSH=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ $DO_PUSH -eq 1 && -z "$COMMIT_MSG" ]]; then
  echo "--push requires -m/--message" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_REPO}" ]]; then
  echo "Source repo not found: ${SOURCE_REPO}" >&2
  exit 1
fi

if [[ ! -d "${TARGET_REPO}/.git" ]]; then
  echo "Target repo is not a git repository: ${TARGET_REPO}" >&2
  exit 1
fi

echo "Syncing from ${SOURCE_REPO}"
echo "Syncing into ${TARGET_REPO}"

mkdir -p "${TARGET_REPO}/examples/baselines"

RSYNC_EXCLUDES=(
  --exclude '__pycache__/'
  --exclude '.ipynb_checkpoints/'
  --exclude '.codex'
  --exclude '*.pt'
  --exclude '*.pth'
  --exclude '*.ckpt'
  --exclude '.DS_Store'
  --exclude 'runs/'
  --exclude 'wandb/'
)

RSYNC_ARGS=(-a --delete --delete-excluded "${RSYNC_EXCLUDES[@]}")
if [[ $DRY_RUN -eq 1 ]]; then
  RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

rsync "${RSYNC_ARGS[@]}" \
  "${SOURCE_REPO}/STPM/" \
  "${TARGET_REPO}/STPM/"

mkdir -p "${TARGET_REPO}/examples/baselines/diffusion_policy"
rsync "${RSYNC_ARGS[@]}" \
  "${SOURCE_REPO}/examples/baselines/diffusion_policy/" \
  "${TARGET_REPO}/examples/baselines/diffusion_policy/"

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "Dry run finished. No files changed."
  exit 0
fi

git -C "${TARGET_REPO}" status --short --branch

if [[ -z "${COMMIT_MSG}" ]]; then
  echo
  echo "Sync finished. No commit created."
  exit 0
fi

git -C "${TARGET_REPO}" add \
  .gitignore \
  STPM \
  examples/baselines/diffusion_policy \
  scripts/sync_from_maniskill.sh

if git -C "${TARGET_REPO}" diff --cached --quiet; then
  echo
  echo "No synced changes to commit."
  exit 0
fi

git -C "${TARGET_REPO}" commit -m "${COMMIT_MSG}"

if [[ $DO_PUSH -eq 1 ]]; then
  CURRENT_BRANCH="$(git -C "${TARGET_REPO}" branch --show-current)"
  if [[ -z "${CURRENT_BRANCH}" ]]; then
    echo "Cannot detect current branch for push." >&2
    exit 1
  fi
  git -C "${TARGET_REPO}" push origin "${CURRENT_BRANCH}"
fi

echo
echo "Done."
