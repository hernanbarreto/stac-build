#!/bin/bash
# STAC Build — THE WAY INTO THE POD, ON EVERY POD (USER 2026-11-28: "El arranque automático (init_pod.sh) tiene que
# dejar el SSH funcionando en cada pod nuevo, sin que yo haga nada").
#
# Run by init_pod.sh at EVERY run, before anything else; it only sets what is missing, so it can be run again at any
# time (bash scripts/pod_access.sh) and leaves what is already right as it is. Its result is one line per service,
# "ssh: …" and "jupyter: …", in the start's log (logs/autostart_latest.log); pod_autostart.sh repeats both lines in
# the container's log (RunPod → the pod → Logs), which can be read when SSH is what failed.
#   · SSH: his key appended to /root/.ssh/authorized_keys when it is not there; /root not writable by group / others,
#     /root/.ssh 700, authorized_keys 600 (sshd's StrictModes refuses the key otherwise); the host keys that are
#     missing (ssh-keygen -A); sshd started when nothing listens on its port.
#   · JupyterLab on :8888 when JUPYTER_PASSWORD is set (the pod template's variable, the one /start.sh reads too),
#     with that password as its token. Without it Jupyter is not started: with no password it would be a root shell
#     open to anyone who has the pod's :8888 address.
# It always exits 0: a failure here is reported, never a reason for the services not to start.

SSH_HOME=/root
SSH_KEY_TYPE=ssh-ed25519
SSH_KEY_BODY=AAAAC3NzaC1lZDI1NTE5AAAAIGMc7XWB+rWYEu9WVVKt0xG8kvUtAGfn4bmETT/gozuR
SSH_KEY_NAME=hernan
SSHD=/usr/sbin/sshd
JUPYTER_PORT=8888
JUPYTER_LOG=/jupyter.log          # the container's disk, where the image's own start writes it

listening() { [ -n "$(ss -ltnH "sport = :$1" 2> /dev/null)" ]; }

# his key in <home>/.ssh/authorized_keys and the permissions sshd asks for; sets KEY_STATE.
# A line carrying his key under a mangled type is removed: the pod's PUBLIC_KEY setting (his RunPod account's SSH
# key, which /start.sh appends to every new pod) held it as "sh-ed25519 …", its first "s" lost (found 2026-11-28).
ensure_key() {
    local home="$1" dir="$1/.ssh" file="$1/.ssh/authorized_keys" kept removed=""
    mkdir -p "$dir" && touch "$file" || return 1
    if awk -v t="$SSH_KEY_TYPE" -v k="$SSH_KEY_BODY" '$2 == k && $1 != t {bad = 1} END {exit !bad}' "$file"; then
        kept="$(awk -v t="$SSH_KEY_TYPE" -v k="$SSH_KEY_BODY" '!($2 == k && $1 != t)' "$file")" || return 1
        if [ -n "$kept" ]; then printf '%s\n' "$kept" > "$file"; else : > "$file"; fi || return 1
        removed=", a broken copy of it removed"
    fi
    if awk -v t="$SSH_KEY_TYPE" -v k="$SSH_KEY_BODY" '$1 == t && $2 == k {found = 1} END {exit !found}' "$file"; then
        KEY_STATE="already there$removed"
    else
        if [ -s "$file" ] && [ -n "$(tail -c 1 "$file")" ]; then echo >> "$file" || return 1; fi
        echo "$SSH_KEY_TYPE $SSH_KEY_BODY $SSH_KEY_NAME" >> "$file" || return 1
        KEY_STATE="added$removed"
    fi
    chmod go-w "$home" && chmod 700 "$dir" && chmod 600 "$file"
}

start_ssh() {
    local err port i how
    if ! ensure_key "$SSH_HOME"; then
        echo "ssh: ERROR — $SSH_HOME/.ssh/authorized_keys could not be written or its permissions set"
        return
    fi
    if [ ! -x "$SSHD" ]; then
        echo "ssh: $SSHD is not installed — installing openssh-server"
        DEBIAN_FRONTEND=noninteractive timeout 600 bash -c \
            'apt-get update -qq && apt-get install -y -qq openssh-server' > /dev/null 2>&1
        if [ ! -x "$SSHD" ]; then echo "ssh: ERROR — openssh-server could not be installed"; return; fi
    fi
    if ! err="$(ssh-keygen -A 2>&1)"; then echo "ssh: ERROR — the host keys could not be generated: $err"; return; fi
    mkdir -p /run/sshd && chmod 755 /run/sshd
    if ! err="$("$SSHD" -t 2>&1)"; then echo "ssh: ERROR — sshd -t refuses the configuration: $err"; return; fi
    port="$("$SSHD" -T 2> /dev/null | awk '$1 == "port" {print $2; exit}')"
    port="${port:-22}"
    how="was running"
    if ! listening "$port"; then
        how="started"
        if command -v service > /dev/null 2>&1; then
            service ssh start > /dev/null 2>&1 || "$SSHD"
        else
            "$SSHD"
        fi
        for ((i = 0; i < 20; i++)); do listening "$port" && break; sleep 0.5; done
    fi
    if listening "$port"; then
        echo "ssh: up — port $port, sshd $how, key '$SSH_KEY_NAME' $KEY_STATE"
    else
        echo "ssh: ERROR — sshd is not listening on port $port"
    fi
}

start_jupyter() {
    local py i
    if [ -z "${JUPYTER_PASSWORD:-}" ]; then
        echo "jupyter: not started — JUPYTER_PASSWORD is not set"
        return
    fi
    if listening "$JUPYTER_PORT"; then
        echo "jupyter: up — port $JUPYTER_PORT was already listening"
        return
    fi
    py=/usr/local/bin/python
    [ -x "$py" ] || py="$(command -v python3)"
    if ! "$py" -c 'import jupyterlab' 2> /dev/null; then
        echo "jupyter: ERROR — JupyterLab is not installed for $py"
        return
    fi
    # the token through the environment (JUPYTER_TOKEN), never on the command line that `ps` shows
    (cd / && JUPYTER_TOKEN="$JUPYTER_PASSWORD" setsid nohup "$py" -m jupyter lab --allow-root --no-browser \
        --port="$JUPYTER_PORT" --ip='*' --FileContentsManager.delete_to_trash=False \
        --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' --ServerApp.allow_origin='*' \
        --ServerApp.root_dir=/ --ServerApp.preferred_dir=/workspace > "$JUPYTER_LOG" 2>&1 < /dev/null &)
    for ((i = 0; i < 120; i++)); do listening "$JUPYTER_PORT" && break; sleep 0.5; done
    if listening "$JUPYTER_PORT"; then
        echo "jupyter: up — port $JUPYTER_PORT, started (log $JUPYTER_LOG)"
    else
        echo "jupyter: ERROR — nothing listens on port $JUPYTER_PORT after 60 s; the end of $JUPYTER_LOG:"
        tail -n 5 "$JUPYTER_LOG" 2> /dev/null | sed 's/^/jupyter:   /'
    fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    start_ssh
    start_jupyter
    exit 0
fi
