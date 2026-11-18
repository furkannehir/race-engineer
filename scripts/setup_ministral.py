"""Verify/download Ministral and the pinned Windows runtime without editing config."""

import argparse

if __package__:
    from .setup_dialogue_comparison import ASSETS, MODELS
    from .setup_lfm3_trial import install_runtime
    from .setup_qwen_gpu import download
else:
    from setup_dialogue_comparison import ASSETS, MODELS
    from setup_lfm3_trial import install_runtime
    from setup_qwen_gpu import download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    print("Verifying Ministral Q4_K_M weights...", flush=True)
    download(MODELS["ministral"], ASSETS)
    print(f"Verifying llama.cpp b12088 ({args.backend})...", flush=True)
    install_runtime(args.backend)
    print("Ministral is ready. Configuration was not changed.", flush=True)


if __name__ == "__main__":
    main()
