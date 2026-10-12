"""Authenticate direct Tailscale connections belonging to the server's owner."""

import ipaddress
import json
import subprocess
from urllib.parse import urlsplit


def local_api(*args):
    result = subprocess.run(
        ["tailscale", *args], check=True, capture_output=True, text=True, timeout=3,
    )
    return json.loads(result.stdout)


class TailscaleAuth:
    def __init__(self):
        status = local_api("status", "--json")
        own = status.get("Self") or {}
        self.owner = own.get("UserID")
        self.addresses = set(own.get("TailscaleIPs") or [])
        self.hostnames = self.addresses | {
            name.lower().rstrip(".") for name in (own.get("DNSName"), own.get("HostName")) if name
        }
        if status.get("BackendState") != "Running" or not self.owner or not self.addresses:
            raise RuntimeError("Tailscale must be connected before enabling Tailscale authentication")
        if own.get("Tags"):
            raise RuntimeError("Tailscale authentication requires a user-owned server")

    def allows(self, connection, peer, host_header):
        # Check the actual socket endpoints, never browser-supplied proxy headers.
        if connection.getsockname()[0] not in self.addresses:
            return False
        try:
            # Reject unrelated domains resolving to our IP (DNS rebinding).
            host_url = urlsplit("//" + host_header)
            if (not host_url.hostname or host_url.hostname.lower().rstrip(".") not in self.hostnames
                    or host_url.port != connection.getsockname()[1] or host_url.username is not None):
                return False
            host, port = peer[:2]
            address = ipaddress.ip_address(host)
            endpoint = f"[{host}]:{port}" if address.version == 6 else f"{host}:{port}"
            identity = local_api("whois", "--json", "--proto=tcp", endpoint)
            node = identity.get("Node") or {}
            user = identity.get("UserProfile") or {}
            node_addresses = {ipaddress.ip_interface(cidr).ip for cidr in node.get("Addresses", [])}
            return (user.get("ID") == self.owner and node.get("User") == self.owner
                    and not node.get("Tags") and address in node_addresses)
        except (OSError, subprocess.SubprocessError, ValueError, TypeError):
            return False
