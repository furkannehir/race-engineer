"""Start the configured local conversation model on a loopback-only llama.cpp server."""

import argparse
import asyncio
from pathlib import Path

from race_engineer.config import ConversationConfig, ConversationRuntimeConfig, load_config
from race_engineer.conversation.runtime import ConversationServer, run_server_until_exit

ROOT = Path(__file__).resolve().parents[1]
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument("--server", type=Path, help="override the configured CPU executable")
    parser.add_argument("--model", type=Path, help="override the configured GGUF path")
    parser.add_argument("--port", type=int, help="override the configured loopback port")
    parser.add_argument("--threads", type=int, help="override the configured CPU thread count")
    args = parser.parse_args()
    config = load_config(args.config)
    runtime_values = config.conversation.runtime.model_dump()
    if args.server is not None:
        runtime_values["cpu_executable_path"] = args.server
        runtime_values["mode"] = "cpu"
    if args.model is not None:
        runtime_values["model_path"] = args.model
    if args.threads is not None:
        runtime_values["threads"] = args.threads
    runtime = ConversationRuntimeConfig.model_validate(runtime_values)
    conversation_values = config.conversation.model_dump()
    conversation_values["port"] = (
        args.port if args.port is not None else config.conversation.port
    )
    conversation_values["runtime"] = runtime
    conversation = ConversationConfig.model_validate(conversation_values)
    server = ConversationServer(
        ROOT,
        conversation,
        quiet=False,
        notify=lambda topic, value: print(f"{topic}: {value}", flush=True),
    )
    try:
        result = asyncio.run(run_server_until_exit(server))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(f"could not start the local runtime: {error}")
    raise SystemExit(result)


if __name__ == "__main__":
    main()
