#!/usr/bin/env python3
"""Connect the offload server to rapid-mlx's public tunnel infrastructure.

Expects the offload server to already be running on --port.
Creates a public URL on rapidmlx.com that you can share.
"""
import secrets
import signal
import sys
import time

from vllm_mlx.share import ws_tunnel

CHAT_FRONTEND = "https://rapid-pro.quicksilverpro.io"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080, help="Local offload server port")
    args = parser.parse_args()

    # Verify local server is up
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=3) as r:
            if r.status != 200:
                print(f"Server on port {args.port} not healthy", file=sys.stderr)
                sys.exit(1)
    except Exception as e:
        print(f"Cannot reach server on port {args.port}: {e}", file=sys.stderr)
        print("Start the offload server first:", file=sys.stderr)
        print(f"  python scripts/expert_offload_mvp.py --serve --port {args.port}", file=sys.stderr)
        sys.exit(1)

    api_key = secrets.token_hex(24)
    tunnel = ws_tunnel.TunnelClient(local_port=args.port)

    print(f"Connecting tunnel to port {args.port}...", file=sys.stderr)
    tunnel_thread = tunnel.run_in_thread()

    if not tunnel.ready_event.wait(timeout=30):
        print(f"Tunnel did not connect within 30s", file=sys.stderr)
        if tunnel.error:
            print(f"  Error: {tunnel.error}", file=sys.stderr)
        sys.exit(1)

    if tunnel.error:
        print(f"Tunnel error: {tunnel.error}", file=sys.stderr)
        sys.exit(1)

    chat_link = f"{CHAT_FRONTEND}/#k={tunnel.tunnel_id}.{api_key}"

    print(f"\n{'='*60}")
    print(f"  Mixtral 8x7B (47B params) — Expert Offloaded")
    print(f"  Running on 32GB Mac mini via SSD expert streaming")
    print(f"{'='*60}")
    print(f"\n  Chat URL: {chat_link}")
    print(f"\n  API URL:  {tunnel.public_url}/v1/chat/completions")
    print(f"\n  Note: ~0.3 tok/s (slow but it works!)")
    print(f"  Press Ctrl-C to stop.\n")

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        while True:
            if tunnel.closed_event.is_set():
                print("Tunnel disconnected.", file=sys.stderr)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping tunnel...")
    finally:
        tunnel.stop()
        if tunnel_thread.is_alive():
            tunnel_thread.join(timeout=5)

if __name__ == "__main__":
    main()
