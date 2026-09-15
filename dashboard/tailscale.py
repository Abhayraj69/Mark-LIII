"""
dashboard/tailscale.py — internet-reachable access to the JARVIS phone
dashboard via Tailscale, a private WireGuard mesh network, instead of a
public tunnel.

Once Tailscale is installed and logged in on BOTH this PC and the phone —
a one-time step only the user can do, since it's tied to their own account
— this PC gets a stable address (its Tailscale IP, or a MagicDNS hostname
if the user's tailnet has that turned on) reachable from the phone from
anywhere both have internet. Nothing is exposed to the public internet:
only devices already on the same tailnet can reach it at all, unlike a
Cloudflare Tunnel (this module's predecessor), which hands out a real
public URL protected only by dashboard/server.py's own login — and which
also depended on a temporary provisioning request to Cloudflare's API that
proved unreliable in practice (timeouts, rate limits, dead links once the
process stopped). Tailscale has no equivalent "spin up a tunnel and hope it
registers in time" step: once it reports an address, that address is
already routable.

This module only ever reads state from an already-running `tailscale`
client on PATH. It never installs Tailscale, runs `tailscale up`, or signs
into an account on the user's behalf — that needs their own login and has
to happen in their own browser.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys


def tailscale_path() -> str | None:
    return shutil.which("tailscale")


def install_hint() -> str:
    if sys.platform == "win32":
        return "winget install --id Tailscale.Tailscale -e"
    if sys.platform == "darwin":
        return "brew install --cask tailscale"
    return "https://tailscale.com/download/linux  (or your distro's package manager)"


def get_status() -> dict:
    """Raw `tailscale status --json` output, or {} if tailscale isn't
    installed or the command fails for any reason."""
    exe = tailscale_path()
    if not exe:
        return {}
    try:
        r = subprocess.run(
            [exe, "status", "--json"],
            capture_output=True, text=True, timeout=8,
        )
        if r.returncode != 0:
            return {}
        return json.loads(r.stdout)
    except Exception:
        return {}


def get_address() -> tuple[str | None, str | None]:
    """(tailscale_ip, magicdns_hostname) for THIS device, or (None, None) if
    tailscale isn't installed, isn't logged in, or has no address yet."""
    status = get_status()
    self_info = status.get("Self") or {}
    ips = self_info.get("TailscaleIPs") or []
    ip = next((a for a in ips if "." in a), None)   # prefer the IPv4 address over IPv6
    dns = (self_info.get("DNSName") or "").rstrip(".") or None
    return ip, dns


def login_state() -> str:
    """One of: 'missing' (not installed), 'needs_login', 'ready', 'unknown'
    (installed but status couldn't be read — e.g. the service isn't running
    yet right after install)."""
    if not tailscale_path():
        return "missing"
    status = get_status()
    if not status:
        return "unknown"
    backend = status.get("BackendState", "")
    if backend == "Running":
        return "ready"
    if backend in ("NeedsLogin", "NeedsMachineAuth", "Stopped"):
        return "needs_login"
    return "unknown"
