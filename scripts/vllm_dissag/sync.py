#!/usr/bin/env python3
"""
Multi-node synchronization utilities for disaggregated inference.

Subcommands:
    barrier  - Wait until all specified nodes have opened their ports (TCP barrier)
               Optionally wait for HTTP health endpoints to return 200
    wait     - Block until a remote port closes (shutdown coordination)
"""

import socket
import time
import threading
import argparse
import sys
import urllib.request
import urllib.error


def is_port_open(ip, port, timeout=2):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((ip, port)) == 0


def check_health(ip, port, path="/health", timeout=2):
    try:
        url = f"http://{ip}:{port}{path}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return getattr(resp, "status", 200) == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def cmd_barrier(args):
    NODE_IPS = [ip.strip() for ip in args.node_ips.split(",") if ip.strip()]
    NODE_PORTS = [int(p.strip()) for p in args.node_ports.split(",") if p.strip()]

    if not NODE_IPS:
        print("Error: NODE_IPS argument is empty or not set.")
        sys.exit(1)

    if len(NODE_PORTS) == 1:
        NODE_PORTS *= len(NODE_IPS)
    elif len(NODE_PORTS) != len(NODE_IPS):
        print("Error: Number of ports must match number of node IPs or be a single port.")
        sys.exit(1)

    server_socket = None

    def open_port():
        nonlocal server_socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((args.local_ip, args.local_port))
        server_socket.listen(5)
        print(f"Port {args.local_port} is now open on {args.local_ip}.")
        while True:
            conn, _ = server_socket.accept()
            conn.close()

    def close_port():
        nonlocal server_socket
        if server_socket:
            server_socket.close()
            print(f"Port {args.local_port} has been closed on {args.local_ip}.")

    if args.enable_port:
        threading.Thread(target=open_port, daemon=True).start()

    if args.wait_for_all_ports:
        start_time = time.time()
        timeout = args.timeout
        while True:
            if timeout > 0:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    not_open = [(ip, port) for ip, port in zip(NODE_IPS, NODE_PORTS)
                                if not is_port_open(ip, port)]
                    print(f"ERROR: Timeout after {timeout}s waiting for ports.", flush=True)
                    for ip, port in not_open:
                        print(f"  - {ip}:{port}", flush=True)
                    sys.exit(1)
            all_open = all(is_port_open(ip, port) for ip, port in zip(NODE_IPS, NODE_PORTS))
            if all_open:
                break
            if timeout > 0:
                remaining = timeout - (time.time() - start_time)
                print(f"Waiting for nodes {list(zip(NODE_IPS, NODE_PORTS))} .. ({remaining:.0f}s remaining)", flush=True)
            else:
                print(f"Waiting for nodes {list(zip(NODE_IPS, NODE_PORTS))} ..", flush=True)
            time.sleep(5)

    if args.wait_for_all_health:
        health_path = args.health_endpoint
        start_time = time.time()
        timeout = args.timeout
        while True:
            if timeout > 0:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    not_ready = [(ip, port) for ip, port in zip(NODE_IPS, NODE_PORTS)
                                 if not check_health(ip, port, health_path)]
                    print(f"ERROR: Timeout after {timeout}s waiting for health endpoints.", flush=True)
                    for ip, port in not_ready:
                        print(f"  - http://{ip}:{port}{health_path}", flush=True)
                    sys.exit(1)
            all_ready = all(check_health(ip, port, health_path) for ip, port in zip(NODE_IPS, NODE_PORTS))
            if all_ready:
                break
            if timeout > 0:
                remaining = timeout - (time.time() - start_time)
                print(f"Waiting for health on {list(zip(NODE_IPS, NODE_PORTS))} ({health_path}) .. ({remaining:.0f}s remaining)", flush=True)
            else:
                print(f"Waiting for health on {list(zip(NODE_IPS, NODE_PORTS))} ({health_path}) ..", flush=True)
            time.sleep(30)

    if args.enable_port:
        time.sleep(30)
        close_port()


def cmd_wait(args):
    print(f"Waiting while port {args.remote_port} on {args.remote_ip} is open...")
    while is_port_open(args.remote_ip, args.remote_port):
        time.sleep(5)
    print(f"Port {args.remote_port} on {args.remote_ip} is now closed.")


def main():
    parser = argparse.ArgumentParser(description="Multi-node synchronization utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bp = subparsers.add_parser("barrier", help="Wait for all nodes to open specified ports.")
    bp.add_argument("--local-ip", required=False)
    bp.add_argument("--local-port", type=int, required=False)
    bp.add_argument("--enable-port", action="store_true")
    bp.add_argument("--node-ips", required=True, help="Comma-separated list of node IPs.")
    bp.add_argument("--node-ports", required=True, help="Comma-separated list of ports to check.")
    bp.add_argument("--timeout", type=int, default=600)
    bp.add_argument("--wait-for-all-ports", action="store_true")
    bp.add_argument("--wait-for-all-health", action="store_true")
    bp.add_argument("--health-endpoint", default="/health")
    bp.set_defaults(func=cmd_barrier)

    wp = subparsers.add_parser("wait", help="Wait while a remote port remains open.")
    wp.add_argument("--remote-ip", required=True)
    wp.add_argument("--remote-port", type=int, required=True)
    wp.set_defaults(func=cmd_wait)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
