#!/bin/bash
# STAC Build — EVERYTHING STARTS BY ITSELF WHEN THE POD STARTS (USER 2026-11-20: "y que todo debe arrancar
# automaticamente al iniciar el pod tambien").
#
# HOW IT IS CALLED. The container's own disk is made new at every pod start: what survives is the volume and the
# pod's settings. The provider's image runs /start.sh as the container's command (nginx, sshd, Jupyter, the
# container's environment written to /etc/rp_environment). THE POD'S START COMMAND — a setting of the pod
# (RunPod → the pod → Edit Pod → "Container Start Command"; saving it restarts the pod) — runs this file beside it:
#
#     bash -c "bash /workspace/stac-build/scripts/pod_autostart.sh & exec /start.sh"
#
# IT CANNOT KEEP THE POD FROM STARTING: it runs in the background of that command (/start.sh is exec'ed whatever
# happens here — no volume, a syntax error in this file), every wait is bounded, and it always exits 0. Without
# that setting nothing changes: init_pod.sh is run by hand, as before.
#
# WHAT IT DOES. Waits for the volume and for /start.sh to have written the environment, gives init_pod.sh THE
# ENVIRONMENT OF THE SSH LOGIN it has always been started from (measured 2026-11-20 on the running services
# against /start.sh's own: the container's variables as a login shell reads them from /etc/rp_environment —
# so without PUBLIC_KEY —, USER / LOGNAME, the locale of /etc/default/locale, conda's shell functions loaded with
# nothing activated, the home directory as the working one; NOT reproduced: SSH_CLIENT / SSH_CONNECTION /
# SSH_TTY and the login shell's LS_COLORS), and runs it ONCE — init_pod.sh refuses a second start too. What it
# prints here goes to the container's log (RunPod's Logs tab): a few lines, never a secret; init_pod.sh's own
# output goes to logs/autostart_<utc>.log (the container's disk when the volume refuses the write).
ROOT="${STAC_AUTOSTART_ROOT:-/workspace/stac-build}"          # (the tests point these two at a stub)
SESSION="${STAC_AUTOSTART_SESSION:-backend}"                   # the tmux session whose existence = started
WAIT_VOLUME_S=180
WAIT_ENV_S=180
CONDA_SH=/workspace/miniforge3/etc/profile.d/conda.sh

say() { echo "[autostart $(date -u +%H:%M:%S)] $*"; }

main() {
    local i log rc user
    # the volume (mounted before the container's command runs: the wait covers a slow mount)
    for ((i = 0; i < WAIT_VOLUME_S; i += 3)); do [ -f "$ROOT/init_pod.sh" ] && break; sleep 3; done
    if [ ! -f "$ROOT/init_pod.sh" ]; then
        say "no $ROOT/init_pod.sh after ${WAIT_VOLUME_S} s — nothing started (by hand: bash $ROOT/init_pod.sh)"
        return 0
    fi
    # one automatic start at a time (the lock lives on the container's disk and goes with the container)
    exec 9> /tmp/stac_autostart.lock
    if ! flock -n 9; then say "another automatic start is running — nothing done"; return 0; fi
    if command -v tmux > /dev/null 2>&1 && tmux has-session -t "$SESSION" 2> /dev/null; then
        say "the services are already started (tmux session '$SESSION' exists) — nothing done"
        return 0
    fi
    # /start.sh writes the container's environment for the login shells AFTER sshd and Jupyter are up; the
    # panes' shells read it from ~/.bashrc
    export HOME="${HOME:-/root}"
    for ((i = 0; i < WAIT_ENV_S; i += 2)); do
        [ -s /etc/rp_environment ] && grep -qs 'source /etc/rp_environment' "$HOME/.bashrc" && break
        sleep 2
    done
    # ── the environment of the SSH login the services have always been started from
    unset PUBLIC_KEY
    if [ -s /etc/rp_environment ]; then
        # shellcheck disable=SC1091
        . /etc/rp_environment
    else
        say "⚠ /etc/rp_environment was not written within ${WAIT_ENV_S} s — starting with the container's environment as it is"
    fi
    user="$(id -un 2> /dev/null || echo root)"
    export USER="$user" LOGNAME="$user" SHELL=/bin/bash TERM=xterm-256color
    if [ -r /etc/default/locale ]; then set -a; . /etc/default/locale; set +a; fi   # (what sshd's login reads)
    if [ -r "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        . "$CONDA_SH"
    fi
    unset OLDPWD
    cd "$HOME" 2> /dev/null || cd /
    # ── init_pod.sh, once, its output in a file
    log="$ROOT/logs/autostart_$(date -u +%Y%m%d_%H%M%S).log"
    mkdir -p "$ROOT/logs" 2> /dev/null
    if ! : 2> /dev/null >> "$log"; then            # the volume refuses the write (its quota): the container's disk
        log="/tmp/stac_autostart_$(date -u +%Y%m%d_%H%M%S).log"
        : >> "$log"
    fi
    ln -sfn "$log" "$ROOT/logs/autostart_latest.log" 2> /dev/null
    say "starting the services (init_pod.sh) — its output: $log"
    env -u STAC_AUTOSTART_ROOT -u STAC_AUTOSTART_SESSION bash "$ROOT/init_pod.sh" >> "$log" 2>&1 < /dev/null 9>&-
    rc=$?
    if [ "$rc" -ne 0 ]; then say "⚠ init_pod.sh ended with code $rc — see $log"; fi
    # the way in (scripts/pod_access.sh), here too: this log can be read when SSH is what failed
    grep -E '^(ssh|jupyter|packages|remote):' "$log" 2> /dev/null | while IFS= read -r line; do say "$line"; done
    if command -v tmux > /dev/null 2>&1; then
        say "tmux sessions: $(tmux list-sessions -F '#{session_name}' 2> /dev/null | tr '\n' ' ')"
    else
        say "⚠ tmux is not installed — no service was started (see $log)"
    fi
    if [ -x "$ROOT/scripts/remote_access.sh" ]; then
        bash "$ROOT/scripts/remote_access.sh" status 2> /dev/null | grep -E "^gate:|^tailscaled:|Funnel on" | head -3 \
            | while IFS= read -r line; do say "$line"; done
    fi
    say "done — the backend loads the chat's model for a few minutes more (logs/latest.log)"
    return 0
}

main "$@"
exit 0
