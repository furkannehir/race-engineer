"""Start the downloaded local Qwen model without changing system execution policies."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data" / "conversation-prototype"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=ASSETS / "llama-b10964-cpu/llama-server.exe")
    parser.add_argument("--model", type=Path, default=ASSETS / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not args.server.is_file() or not args.model.is_file():
        parser.error("local runtime or model missing; see docs/conversation.md")
    if not 1 <= args.port <= 65535 or not 1 <= args.threads <= 64:
        parser.error("port must be 1-65535 and threads must be 1-64")
    try:
        result = subprocess.run(
            [
                str(args.server.resolve()),
                "--model",
                str(args.model.resolve()),
                "--alias",
                "Qwen3-4B-Instruct-2507",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--cors-origins",
                f"http://127.0.0.1:{args.port}",
                "--ctx-size",
                "8192",
                "--parallel",
                "1",
                "--threads",
                str(args.threads),
                "--n-gpu-layers",
                "0",
            ],
            check=False,
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except OSError:
        parser.error("could not start the local runtime; check the runtime path and dependencies")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
