#!/bin/bash
# Restart the backend that runs in the tmux session `backend` — and NEVER hang doing it.
#
#     bash scripts/restart_backend.sh            exit 0 = the API answers, 1 = it does not
#
# What went wrong before (2026-11-14, USER: "a vos no te esta funcionando reiniciarlo, revisa que
# estas haciendo mal … sino te quedas colgado y perdemos tiempo"): ONE Ctrl-C, a fixed sleep, then
# the start command typed into the terminal, then a wait with no bound. With a viewer connected
# uvicorn does not end on the first interrupt (it waits for its connections), so it still owned
# the terminal when the start command was typed: the line was lost, the backend never came back
# and the caller waited for it for ever.
#
# Here every step is checked and bounded:
#   1. Ctrl-C, then wait until the terminal is back at the shell;
#   2. still running → a second Ctrl-C (uvicorn's own force quit); still running → SIGKILL of the
#      terminal's foreground process (its vLLM / tools children die with it: PR_SET_PDEATHSIG);
#   3. the start command is typed ONLY at the shell's prompt, on a cleared line;
#   4. the API is awaited for a bounded time; if the server died at its start the wait ends at
#      once and the terminal's last lines are printed.
set -u

SESSION="${STAC_BACKEND_TMUX:-backend}"
STAC_ROOT="/workspace/stac-build"
PORT=8765
STOP_WAIT_S=20          # after each interrupt
KILL_WAIT_S=10
START_WAIT_S=240        # the API answers ~10-20 s after the start; a bound, not a sleep

foreground() { tmux display-message -p -t "$SESSION" '#{pane_current_command}' 2>/dev/null; }
at_shell() { case "$(foreground)" in bash|sh|zsh) return 0 ;; *) return 1 ;; esac; }
wait_shell() {                       # $1 seconds; 0 when the terminal is back at the shell
    local n=$1
    while (( n-- > 0 )); do
        at_shell && return 0
        sleep 1
    done
    at_shell
}
api_up() { curl -sk -m 3 -o /dev/null "https://localhost:$PORT/api/pipelines/active"; }
tail_of_terminal() { tmux capture-pane -t "$SESSION" -p -S -40 | tail -25 | cut -c1-220; }

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[restart] no tmux session '$SESSION' — start the backend there: bash scripts/start.sh"
    exit 1
fi

# ── 1-2. stop what runs ─────────────────────────────────────────────────────────────────────
if ! at_shell; then
    tmux send-keys -t "$SESSION" C-c
    if ! wait_shell "$STOP_WAIT_S"; then
        echo "[restart] still running ${STOP_WAIT_S}s after the interrupt — forcing (second Ctrl-C)"
        tmux send-keys -t "$SESSION" C-c
        if ! wait_shell "$STOP_WAIT_S"; then
            shell_pid=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
            victims=$(pgrep -P "$shell_pid" | tr '\n' ' ')
            echo "[restart] still running — SIGKILL of the terminal's foreground process: $victims"
            for v in $victims; do kill -9 "$v" 2>/dev/null; done
            if ! wait_shell "$KILL_WAIT_S"; then
                echo "[restart] the terminal did not come back to the shell — nothing was started:"
                tail_of_terminal
                exit 1
            fi
        fi
    fi
fi
if api_up; then
    echo "[restart] the terminal is at the shell but port $PORT still answers: another backend runs"
    echo "          outside this tmux session — nothing was started"
    exit 1
fi

# ── 3. start, from the prompt ───────────────────────────────────────────────────────────────
tmux send-keys -t "$SESSION" C-u
tmux send-keys -t "$SESSION" "cd $STAC_ROOT && bash scripts/start.sh" Enter

# ── 4. wait for the API, bounded ────────────────────────────────────────────────────────────
t0=$(date +%s)
sleep 2                                   # the shell takes the line; then python owns the terminal
while (( $(date +%s) - t0 < START_WAIT_S )); do
    if api_up; then
        echo "[restart] the backend answers ($(( $(date +%s) - t0 )) s)"
        exit 0
    fi
    if at_shell && (( $(date +%s) - t0 > 8 )); then
        echo "[restart] the server ended at its start:"
        tail_of_terminal
        exit 1
    fi
    sleep 2
done
echo "[restart] the API did not answer within ${START_WAIT_S}s:"
tail_of_terminal
exit 1
