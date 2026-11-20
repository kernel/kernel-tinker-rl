#!/bin/bash
# Runs Pi in a container that cannot reach the GitHub App private key.
#
#   pi-agent-sandbox.sh setup        start the token broker, install the gh and git wrappers, and
#                                    prepare the agent's home with a copy of the checkout
#   pi-agent-sandbox.sh exec CMD...  run CMD in the agent container, in its copy of the checkout
#   pi-agent-sandbox.sh teardown     stop the agent container, revoke the token, and stop the broker
#
# The runner user has passwordless sudo and Docker, and the runner keeps every job secret, the
# private key included, in memory for the whole job. Any process on this VM that gets code run as
# the runner user can therefore reach the key, and much of the hosted image is world-writable, so a
# second Unix user is not enough. The agent runs in a container instead, as the unprivileged
# pi-agent user with no capabilities, no Docker socket and no view of the host's processes. The
# host's /usr, /opt and /etc are mounted read-only so the job's tools work, and the agent's home is
# the only host path it can write. It reaches GitHub through a broker on the Docker bridge that
# keeps the key in memory and serves installation tokens limited to GITHUB_APP_REPOSITORY and
# GITHUB_APP_PERMISSIONS.
#
# setup needs GITHUB_APP_ID, GITHUB_APP_INSTALLATION_ID, GITHUB_APP_PRIVATE_KEY,
# GITHUB_APP_REPOSITORY and GITHUB_APP_PERMISSIONS (a JSON permissions object). exec passes the
# container only the variables named in PI_AGENT_ENV, plus the model credentials that
# kernel/security-workflows' llm-credentials action keeps refreshed, copied into the agent's home.
set -euo pipefail

agent=pi-agent
agent_home=/home/$agent
agent_credentials=$agent_home/.pi/credentials
image=pi-agent-sandbox
bin_dir=/usr/local/libexec/github-app-auth
state=$RUNNER_TEMP/github-app-broker
port=8377
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bridge_ip() {
  ip -4 -o addr show docker0 | awk '{ split($4, a, "/"); print a[1] }'
}

as_agent() {
  sudo -u "$agent" env -i HOME="$agent_home" PATH=/usr/bin:/bin "$@"
}

install_wrappers() {
  local token_url=$1 tmp
  tmp=$(mktemp -d)

  cat > "$tmp/gh" <<'EOF'
#!/bin/bash
set -euo pipefail
token="$(@CURL@ -fsS "@TOKEN_URL@")"
printf '::add-mask::%s\n' "$token" >&2
if [ "${1:-}" = "pr" ] && [ "${2:-}" = "checks" ]; then
  exec env GH_TOKEN="$GITHUB_CHECKS_TOKEN" GITHUB_TOKEN="$GITHUB_CHECKS_TOKEN" "@REAL_GH@" "$@"
fi
if [ "${1:-}" = "pr" ] && [ "${2:-}" = "merge" ]; then
  pr_number="${3:-}"
  if [ -z "$pr_number" ] || [[ "$pr_number" = -* ]]; then
    echo "A pull request number is required for the BugBot merge gate" >&2
    exit 1
  fi
  head_sha="$(env GH_TOKEN="$token" GITHUB_TOKEN="$token" "@REAL_GH@" pr view "$pr_number" --json headRefOid --jq .headRefOid)"
  check_runs="$(env GH_TOKEN="$GITHUB_CHECKS_TOKEN" GITHUB_TOKEN="$GITHUB_CHECKS_TOKEN" "@REAL_GH@" api "repos/$GITHUB_REPOSITORY/commits/$head_sha/check-runs?filter=latest&per_page=100")"
  bugbot_conclusion="$(@JQ@ -r '[.check_runs[] | select(.name == "Cursor Bugbot" and .status == "completed")] | sort_by(.completed_at) | last | .conclusion // ""' <<<"$check_runs")"
  if [ "$bugbot_conclusion" != "success" ]; then
    echo "Cursor Bugbot is not green for $head_sha (conclusion: ${bugbot_conclusion:-missing})" >&2
    exit 1
  fi
  incomplete_checks="$(@JQ@ -r '[.check_runs[] | select(.status != "completed")] | map(.name) | join(", ")' <<<"$check_runs")"
  failed_checks="$(@JQ@ -r '[.check_runs[] | select(.status == "completed" and (.conclusion | IN("success", "neutral", "skipped") | not))] | map("\(.name)=\(.conclusion)") | join(", ")' <<<"$check_runs")"
  if [ -n "$incomplete_checks" ] || [ -n "$failed_checks" ]; then
    echo "Pull request checks are not green for $head_sha (incomplete: ${incomplete_checks:-none}; failed: ${failed_checks:-none})" >&2
    exit 1
  fi
  commit_status="$(env GH_TOKEN="$GITHUB_CHECKS_TOKEN" GITHUB_TOKEN="$GITHUB_CHECKS_TOKEN" "@REAL_GH@" api "repos/$GITHUB_REPOSITORY/commits/$head_sha/status")"
  status_count="$(@JQ@ -r .total_count <<<"$commit_status")"
  status_state="$(@JQ@ -r .state <<<"$commit_status")"
  if [ "$status_count" -gt 0 ] && [ "$status_state" != "success" ]; then
    echo "Commit statuses are not green for $head_sha (state: $status_state)" >&2
    exit 1
  fi
fi
exec env GH_TOKEN="$token" GITHUB_TOKEN="$token" "@REAL_GH@" "$@"
EOF

  cat > "$tmp/git-credential-github-app" <<'EOF'
#!/bin/bash
set -euo pipefail
[ "${1:-}" = "get" ] || exit 0
protocol=""
host=""
while IFS='=' read -r key value; do
  case "$key" in
    protocol) protocol="$value" ;;
    host) host="$value" ;;
  esac
done
[ "$protocol" = "https" ] && [ "$host" = "github.com" ] || exit 0
printf 'username=x-access-token\n'
printf 'password=%s\n' "$(@CURL@ -fsS "@TOKEN_URL@")"
EOF

  cat > "$tmp/trigger-bugbot" <<'EOF'
#!/bin/bash
set -euo pipefail
pr_number="${1:-}"
if [ -z "$pr_number" ] || [[ "$pr_number" = -* ]]; then
  echo "A pull request number is required" >&2
  exit 1
fi
pr_url="$(@BIN_DIR@/gh pr view "$pr_number" --json url --jq .url)"
response="$(@CURL@ --silent --show-error --fail-with-body \
  --request POST \
  --url https://api.cursor.com/bugbot/review \
  --user "$CURSOR_API_KEY:" \
  --header 'Content-Type: application/json' \
  --data "$(@JQ@ -cn --arg prUrl "$pr_url" '{prUrl: $prUrl}')")"
@JQ@ -e '.outcome == "success"' <<<"$response" >/dev/null
@JQ@ -r '"Queued Cursor Bugbot review (request_id: \(.request_id))"' <<<"$response"
EOF

  sed -i "s|@TOKEN_URL@|$token_url|g; s|@REAL_GH@|$(command -v gh)|g; s|@CURL@|$(command -v curl)|g; s|@JQ@|$(command -v jq)|g; s|@BIN_DIR@|$bin_dir|g" "$tmp"/*
  sudo install -d -o root -g root -m 755 "$bin_dir"
  sudo install -o root -g root -m 755 "$tmp/gh" "$tmp/git-credential-github-app" "$tmp/trigger-bugbot" "$bin_dir/"
  rm -rf "$tmp"
}

# An image with nothing in it but the merged-/usr symlinks; exec mounts the host's userland.
build_image() {
  local root
  root=$(mktemp -d)
  mkdir -p "$root"/{usr,etc,opt,home,root,proc,sys,dev,run,tmp,var/tmp}
  chmod 1777 "$root/tmp" "$root/var/tmp"
  ln -s usr/bin "$root/bin"
  ln -s usr/sbin "$root/sbin"
  ln -s usr/lib "$root/lib"
  ln -s usr/lib64 "$root/lib64"
  tar -C "$root" -c . | docker import - "$image" >/dev/null
  rm -rf "$root"
}

setup() {
  : "${GITHUB_APP_PRIVATE_KEY:?}" "${GITHUB_APP_REPOSITORY:?}" "${GITHUB_APP_PERMISSIONS:?}"
  local private_key=$GITHUB_APP_PRIVATE_KEY host token_url
  unset GITHUB_APP_PRIVATE_KEY
  host=$(bridge_ip)
  token_url=http://$host:$port/token

  mkdir -p "$state"
  chmod 700 "$state"
  printf '%s' "$private_key" | GITHUB_APP_BROKER_HOST=$host GITHUB_APP_BROKER_PORT=$port \
    setsid node "$script_dir/github-app-token-broker.mjs" >>"$state/broker.log" 2>&1 &
  echo "$!" > "$state/broker.pid"
  unset private_key
  for _ in $(seq 1 30); do
    curl -fsS -o /dev/null "$token_url" && break
    sleep 1
  done
  if ! curl -fsS -o /dev/null "$token_url"; then
    cat "$state/broker.log" >&2
    echo "::error::The GitHub App token broker did not start" >&2
    exit 1
  fi
  cat "$state/broker.log"

  install_wrappers "$token_url"
  build_image

  sudo useradd --create-home --shell /bin/bash "$agent"
  sudo chmod 700 "$agent_home"
  sudo cp -a "$GITHUB_WORKSPACE" "$agent_home/repo"
  sudo chown -R "$agent:$agent" "$agent_home/repo"
  as_agent mkdir -p "$agent_home/.pi/agent" "$agent_credentials" "$agent_home/.npm-global"
  as_agent git -C "$agent_home/repo" config --local --unset-all http.https://github.com/.extraheader || true
  as_agent git -C "$agent_home/repo" config --local --unset-all credential.helper || true
  as_agent git -C "$agent_home/repo" config --local credential.https://github.com.helper "$bin_dir/git-credential-github-app"
  as_agent git -C "$agent_home/repo" config --local user.name "kernel-internal[bot]"
  as_agent git -C "$agent_home/repo" config --local user.email "260533166+kernel-internal[bot]@users.noreply.github.com"

  echo "PI_AGENT_HOME=$agent_home" >> "$GITHUB_ENV"
  echo "$bin_dir" >> "$GITHUB_PATH"
}

# The llm-credentials action refreshes model credentials as the runner user, in files only that
# user can read. Copy each one into the agent's home, as the agent, whenever the action replaces it.
declare -A copied=()
copy_credential() {
  local source=$1 target=$2 stamp
  [ -n "$source" ] && [ -f "$source" ] || return 0
  stamp=$(stat -c '%i %Y %s' "$source")
  [ "${copied[$target]:-}" = "$stamp" ] && return 0
  # shellcheck disable=SC2024 # the runner user reads the source; the agent writes the copy
  sudo -u "$agent" /bin/sh -c 'cat > "$1.tmp" && mv -f "$1.tmp" "$1"' sh "$target" < "$source"
  copied[$target]=$stamp
}

sync_credentials() {
  copy_credential "${ANTHROPIC_IDENTITY_TOKEN_FILE:-}" "$agent_credentials/anthropic-identity-token"
  copy_credential "${OPENROUTER_ACCESS_TOKEN_FILE:-}" "$agent_credentials/openrouter-access-token"
  copy_credential "${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}/auth.json" "$agent_home/.pi/agent/auth.json"
}

run_as_agent() {
  local args=(
    --rm --name "$agent"
    --user "$(id -u "$agent"):$(id -g "$agent")"
    --cap-drop ALL --security-opt no-new-privileges
    --volume /usr:/usr:ro --volume /opt:/opt:ro --volume /etc:/etc:ro
    --volume "$agent_home:$agent_home"
    --workdir "$agent_home/repo"
    --env HOME="$agent_home" --env USER="$agent" --env LOGNAME="$agent" --env SHELL=/bin/bash
    --env LANG=C.UTF-8 --env CI=true --env GITHUB_ACTIONS=true
    --env GITHUB_REPOSITORY="$GITHUB_REPOSITORY" --env GIT_TERMINAL_PROMPT=0
    --env NPM_CONFIG_PREFIX="$agent_home/.npm-global"
    --env PATH="$bin_dir:$agent_home/.npm-global/bin:$agent_home/.local/bin:$PATH"
  )
  local name
  # Values come from this process's environment rather than the command line.
  for name in ${PI_AGENT_ENV:-}; do
    args+=(--env "$name")
  done
  if [ -n "${ANTHROPIC_IDENTITY_TOKEN_FILE:-}" ]; then
    args+=(
      --env ANTHROPIC_FEDERATION_RULE_ID --env ANTHROPIC_ORGANIZATION_ID
      --env ANTHROPIC_SERVICE_ACCOUNT_ID
      --env ANTHROPIC_IDENTITY_TOKEN_FILE="$agent_credentials/anthropic-identity-token"
    )
  fi
  if [ -n "${OPENROUTER_ACCESS_TOKEN_FILE:-}" ]; then
    args+=(--env OPENROUTER_ACCESS_TOKEN_FILE="$agent_credentials/openrouter-access-token")
  fi

  sync_credentials
  (while sleep 2; do sync_credentials || true; done) &
  relay=$!
  trap 'kill "$relay" 2>/dev/null || true' EXIT
  docker run "${args[@]}" "$image" "$@"
}

teardown() {
  docker rm --force "$agent" >/dev/null 2>&1 || true
  "$bin_dir/gh" api --method DELETE /installation/token || true
  if [ -f "$state/broker.pid" ]; then
    kill "$(cat "$state/broker.pid")" 2>/dev/null || true
  fi
  if [ -f "$state/broker.log" ]; then
    cat "$state/broker.log"
  fi
}

case "${1:-}" in
  setup) setup ;;
  exec) shift; run_as_agent "$@" ;;
  teardown) teardown ;;
  *) echo "usage: $0 setup | exec CMD... | teardown" >&2; exit 2 ;;
esac
