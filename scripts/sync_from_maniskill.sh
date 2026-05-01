#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_REPO="${SOURCE_REPO:-/home/hebu/code/ManiSkill}"

COMMIT_MSG=""
DO_PUSH=0

usage() {
  cat <<'EOF'
Usage:
  sync_from_maniskill.sh
  sync_from_maniskill.sh -m "your commit message"
  sync_from_maniskill.sh -m "your commit message" --push

Options:
  -m, --message   Commit message. If omitted, only sync files.
  --push          Push after commit. Must be used with -m/--message.

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
  --exclude '*.pt'
  --exclude '*.pth'
  --exclude '*.ckpt'
  --exclude '.DS_Store'
)

rsync -a "${RSYNC_EXCLUDES[@]}" \
  "${SOURCE_REPO}/STPM" \
  "${TARGET_REPO}/"

rsync -a "${RSYNC_EXCLUDES[@]}" \
  "${SOURCE_REPO}/examples/baselines/diffusion_policy" \
  "${TARGET_REPO}/examples/baselines/"

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
  git -C "${TARGET_REPO}" push origin main
fi

echo
echo "Done."
