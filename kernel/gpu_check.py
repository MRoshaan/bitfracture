"""BitFracture Phase 0 smoke test: confirm Kaggle T4 GPU + torch CUDA work.

Writes all output under /kaggle/working/ (the only path Kaggle CLI downloads).
"""

import sys

print("Python:", sys.version.split()[0])


def main() -> None:
    try:
        import torch
    except ImportError:
        print("torch not preinstalled in this image; installing CPU build for check...")
        import subprocess

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torch",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
        import torch

    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
        a = torch.randn(2000, 2000, device="cuda")
        b = torch.randn(2000, 2000, device="cuda")
        c = a @ b
        print("matmul ok, sample:", float(c[0, 0]))
        print("RESULT: GPU_OK")
    else:
        print("RESULT: NO_GPU")


if __name__ == "__main__":
    main()
