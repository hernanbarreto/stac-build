#!/bin/bash
# CHECK — the pod starts with no error (USER 2026-11-28: "el pod debe arrancar siempre sin errores"): SSH (and
# JupyterLab when JUPYTER_PASSWORD is set — "El arranque automático (init_pod.sh) tiene que dejar el SSH funcionando
# en cada pod nuevo, sin que yo haga nada"), the start's packages, the Electron desktop and noVNC, the public page.
# Read only. Closed = every line below OK.
ROOT=/workspace/stac-build
KEY_TYPE=ssh-ed25519
KEY_BODY=AAAAC3NzaC1lZDI1NTE5AAAAIGMc7XWB+rWYEu9WVVKt0xG8kvUtAGfn4bmETT/gozuR
FILE=/root/.ssh/authorized_keys
bad=0
ok()  { echo "OK      $*"; }
no()  { echo "NOT OK  $*"; bad=1; }
check() { local what="$1"; shift; if "$@" > /dev/null 2>&1; then ok "$what"; else no "$what"; fi; }

cd "$ROOT" || { echo "NOT CLOSED"; exit 1; }
check "scripts/pod_access.sh, init_pod.sh, pod_autostart.sh: bash syntax" \
    bash -c 'bash -n scripts/pod_access.sh && bash -n init_pod.sh && bash -n scripts/pod_autostart.sh'
check "scripts/pod_access.sh and pod_autostart.sh committed as they are" \
    git diff --quiet HEAD -- scripts/pod_access.sh scripts/pod_autostart.sh
call=$(git show HEAD:init_pod.sh | grep -n 'scripts/pod_access.sh$' | head -1 | cut -d: -f1)
guard=$(git show HEAD:init_pod.sh | grep -n 'tmux has-session -t backend' | head -1 | cut -d: -f1)
check "init_pod.sh (committed) runs pod_access.sh first, before its one-start guard" \
    test -n "$call" -a -n "$guard" -a "${call:-0}" -lt "${guard:-0}"
check "init_pod.sh (working copy) runs it the same way" grep -q '^bash /workspace/stac-build/scripts/pod_access.sh$' init_pod.sh
check "pod_autostart.sh repeats the ssh: / jupyter: / packages: / remote: lines in the container's log" \
    grep -q "grep -E '^(ssh|jupyter|packages|remote):'" scripts/pod_autostart.sh

check "1. $FILE holds the key '$KEY_TYPE … hernan'" \
    awk -v t="$KEY_TYPE" -v k="$KEY_BODY" '$1 == t && $2 == k {f = 1} END {exit !f}' "$FILE"
check "1. no line of $FILE carries that key under another type" \
    awk -v t="$KEY_TYPE" -v k="$KEY_BODY" '$2 == k && $1 != t {b = 1} END {exit b}' "$FILE"
check "2. /root not writable by group or others ($(stat -c %a /root))" \
    bash -c '[ $(( 0$(stat -c %a /root) & 022 )) -eq 0 ]'
check "2. /root/.ssh is 700 ($(stat -c %a /root/.ssh))" test "$(stat -c %a /root/.ssh)" = 700
check "2. $FILE is 600 ($(stat -c %a "$FILE"))" test "$(stat -c %a "$FILE")" = 600
check "3. host keys rsa, ecdsa, ed25519 present" \
    test -s /etc/ssh/ssh_host_rsa_key -a -s /etc/ssh/ssh_host_ecdsa_key -a -s /etc/ssh/ssh_host_ed25519_key
check "3. sshd -t accepts the configuration" /usr/sbin/sshd -t
port=$(/usr/sbin/sshd -T 2> /dev/null | awk '$1 == "port" {print $2; exit}')
check "3. sshd listening on port ${port:-22}" bash -c "[ -n \"\$(ss -ltnH 'sport = :${port:-22}')\" ]"
pub=$(mktemp) && echo "$KEY_TYPE $KEY_BODY hernan" > "$pub"
check "3. sshd accepts the key for root (the server's answer to the public key)" bash -c \
    "timeout 20 ssh -v -o BatchMode=yes -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=no \
     -o UserKnownHostsFile=/dev/null -i '$pub' -p ${port:-22} root@127.0.0.1 true 2>&1 | grep -q 'Server accepts key'"
rm -f "$pub"
check "3. the start's last log says 'ssh: up'" grep -q '^ssh: up' "$(readlink -f logs/autostart_latest.log)"
if [ -n "${JUPYTER_PASSWORD:-}" ]; then
    check "4. JupyterLab listening on 8888" bash -c "[ -n \"\$(ss -ltnH 'sport = :8888')\" ]"
    check "4. JupyterLab refuses a request without the password" \
        bash -c '[ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8888/api/status)" = 403 ]'
else
    ok "4. JUPYTER_PASSWORD is not set on this pod: JupyterLab not started, as ordered"
fi

eval "$(sed -n '/^START_PKGS=/,/libxkbcommon0"$/p' init_pod.sh)"
pkg_installed() { dpkg-query -W -f='${Status}' "$1" 2> /dev/null | grep -q 'install ok installed'; }
miss=""; for p in $START_PKGS; do pkg_installed "$p" || pkg_installed "${p}t64" || miss="$miss $p"; done
check "packages of init_pod.sh's list installed${miss:+ (missing:$miss)}" test -z "$miss" -a -n "$START_PKGS"
check "Electron finds every library it loads" \
    bash -c "! ldd ui/node_modules/electron/dist/electron | grep -q 'not found'"
check "Xvfb running" pgrep -x Xvfb
check "x11vnc running" pgrep -x x11vnc
check "noVNC page answers on 6080" bash -c '[ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:6080/vnc.html)" = 200 ]'
check "Electron running" bash -c 'pgrep -f "electron/dist/electro[n]" > /dev/null'
check "Electron not on the card" bash -c '! nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi electron'
gk=$(grep -E '^(export )?STAC_GATE_KEY=' .env | head -1 | sed -E 's/^(export )?STAC_GATE_KEY=//; s/^"//; s/"$//')
host=$(bash scripts/remote_access.sh link 2> /dev/null | sed -E 's#^https://([^/]+)/.*#\1#')
jar=$(mktemp)
check "the public address: the secret link answers 302" bash -c \
    "[ \"\$(curl -4 -s -m 30 -c '$jar' -o /dev/null -w %{http_code} 'https://$host/__gate?k=$gk')\" = 302 ]"
check "the public address: the page answers 200 with its cookie" bash -c \
    "[ \"\$(curl -4 -s -m 30 -b '$jar' -o /dev/null -w %{http_code} 'https://$host/')\" = 200 ]"
check "the public address: refused without the cookie (403)" bash -c \
    "[ \"\$(curl -4 -s -m 30 -o /dev/null -w %{http_code} 'https://$host/')\" = 403 ]"
rm -f "$jar"

[ "$bad" = 0 ] && echo "CLOSED" || echo "NOT CLOSED"
