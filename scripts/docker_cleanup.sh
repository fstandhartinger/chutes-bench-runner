#!/usr/bin/env bash
# Periodic Docker cleanup for bench-runner server.
# Removes unused images older than 24h and dangling build cache.
# Preserves Sandy runtime images, which are launched on demand and may have no
# long-lived container references between benchmark runs.
# Intended to run via systemd timer (docker-cleanup.timer).
set -euo pipefail

LOG_TAG="docker-cleanup"
PROTECTED_REPO_PREFIX="sandy-runtime:"

log() { logger -t "$LOG_TAG" "$*"; echo "$(date -Iseconds) $*"; }

log "Starting Docker cleanup"

CUTOFF_EPOCH="$(date -d '24 hours ago' +%s)"

mapfile -t USED_IMAGE_IDS < <(
  docker ps -aq | xargs -r docker inspect --format '{{.Image}}' | sort -u
)

is_used_image_id() {
  local image_id="$1"
  local used_id
  for used_id in "${USED_IMAGE_IDS[@]}"; do
    if [[ "$used_id" == "$image_id" ]]; then
      return 0
    fi
  done
  return 1
}

preserved_images=()
removed_images=()

while IFS='|' read -r image_id created repo_tags; do
  [[ -z "$image_id" ]] && continue

  if is_used_image_id "$image_id"; then
    continue
  fi

  created_epoch="$(date -d "$created" +%s 2>/dev/null || echo 0)"
  if [[ "$created_epoch" == "0" ]] || (( created_epoch > CUTOFF_EPOCH )); then
    continue
  fi

  protected=false
  if [[ -n "$repo_tags" ]]; then
    IFS=',' read -r -a tags <<< "$repo_tags"
    for tag in "${tags[@]}"; do
      if [[ "$tag" == "${PROTECTED_REPO_PREFIX}"* ]]; then
        protected=true
        preserved_images+=("$tag")
        break
      fi
    done
  fi

  if [[ "$protected" == true ]]; then
    continue
  fi

  if docker image rm "$image_id" >/dev/null 2>&1; then
    removed_images+=("$image_id")
  fi
done < <(
  docker image inspect $(docker image ls -q --no-trunc | sort -u) \
    --format '{{.Id}}|{{.Created}}|{{if .RepoTags}}{{join .RepoTags ","}}{{end}}'
)

if ((${#removed_images[@]} > 0)); then
  log "Removed ${#removed_images[@]} stale image(s): ${removed_images[*]}"
else
  log "Removed 0 stale image(s)"
fi

if ((${#preserved_images[@]} > 0)); then
  log "Preserved protected runtime image(s): ${preserved_images[*]}"
fi

# Prune dangling build cache older than 7 days
BUILD_RECLAIMED=$(docker builder prune --filter "until=168h" -f 2>&1 | grep -oP 'reclaimed .*' || echo "nothing")
log "Build cache prune: $BUILD_RECLAIMED"

# Report current state
DISK_PCT=$(df / --output=pcent | tail -1 | tr -d ' %')
DOCKER_SIZE=$(docker system df --format '{{.Type}}\t{{.Size}}' 2>/dev/null | paste -sd ', ')
log "Disk usage: ${DISK_PCT}% | Docker: $DOCKER_SIZE"

log "Cleanup complete"
