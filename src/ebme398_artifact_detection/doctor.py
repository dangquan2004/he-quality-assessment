from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .model_bundle import resolve_model_bundle


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    summary: str
    fix: str | None = None


def _run_command(cmd: list[str], *, cwd: str | Path | None = None, timeout: int = 30) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        message = getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc)
        return False, message.strip()
    output = completed.stdout.strip() or completed.stderr.strip()
    return True, output


def _default_hf_token_path() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "token"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "huggingface" / "token"
    return Path.home() / ".cache" / "huggingface" / "token"


def check_python_runtime() -> CheckResult:
    version = sys.version_info
    summary = f"{version.major}.{version.minor}.{version.micro} ({sys.executable})"
    if version < (3, 10) or version >= (3, 12):
        return CheckResult(
            name="Python runtime",
            ok=False,
            summary=f"{summary} is outside the supported range for the full inference stack",
            fix=(
                "Use Python 3.10 or 3.11 for this repo.\n"
                "TRIDENT currently needs to be installed into the same environment and does not support Python 3.12+."
            ),
        )
    return CheckResult(name="Python runtime", ok=True, summary=summary)


def check_openslide() -> CheckResult:
    try:
        import openslide
    except Exception as exc:
        return CheckResult(
            name="OpenSlide",
            ok=False,
            summary=f"openslide import failed: {exc}",
            fix=(
                "Install the OpenSlide system library first.\n"
                "macOS: brew install openslide\n"
                "Ubuntu/Debian: sudo apt-get install libopenslide-dev openslide-tools"
            ),
        )
    library_version = getattr(openslide, "__library_version__", "unknown")
    python_version = getattr(openslide, "__version__", "unknown")
    return CheckResult(
        name="OpenSlide",
        ok=True,
        summary=f"openslide-python {python_version}, OpenSlide library {library_version}",
    )


def check_vips() -> CheckResult:
    resolved = shutil.which("vips")
    if resolved is None:
        return CheckResult(
            name="libvips",
            ok=False,
            summary="vips executable not found on PATH",
            fix=(
                "Install libvips.\n"
                "macOS: brew install vips\n"
                "Ubuntu/Debian: sudo apt-get install libvips-tools"
            ),
        )
    ok, output = _run_command(["vips", "--version"])
    if not ok:
        return CheckResult(
            name="libvips",
            ok=False,
            summary=f"found vips at {resolved}, but version check failed: {output}",
            fix=(
                "Reinstall libvips and make sure `vips --version` works.\n"
                "macOS: brew install vips\n"
                "Ubuntu/Debian: sudo apt-get install libvips-tools"
            ),
        )
    return CheckResult(name="libvips", ok=True, summary=f"{resolved} ({output})")


def check_trident(trident_dir: str | Path) -> CheckResult:
    trident_dir = Path(trident_dir).expanduser().resolve()
    script = trident_dir / "run_batch_of_slides.py"
    if not script.exists():
        return CheckResult(
            name="TRIDENT",
            ok=False,
            summary=f"TRIDENT entrypoint not found at {script}",
            fix=(
                "Clone and install TRIDENT into the same active Python environment.\n"
                "git clone https://github.com/mahmoodlab/TRIDENT.git external/TRIDENT\n"
                "cd external/TRIDENT\n"
                "python -m pip install -e ."
            ),
        )
    import_check = (
        "import sys; "
        f"sys.path.insert(0, {str(trident_dir)!r}); "
        "import trident; "
        "from trident import Processor; "
        "from trident.patch_encoder_models.load import encoder_factory; "
        "from trident.segmentation_models.load import segmentation_model_factory; "
        "print(trident.__file__)"
    )
    ok, output = _run_command([sys.executable, "-c", import_check], cwd=trident_dir, timeout=120)
    if not ok:
        return CheckResult(
            name="TRIDENT",
            ok=False,
            summary=f"TRIDENT import check failed under {sys.executable}: {output}",
            fix=(
                "Install TRIDENT into the same active Python environment as this repo.\n"
                "cd {trident_dir}\n"
                "python -m pip install -e ."
            ).format(trident_dir=trident_dir),
        )
    return CheckResult(name="TRIDENT", ok=True, summary=f"imports cleanly from {trident_dir}")


def check_hugging_face_auth() -> CheckResult:
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token_path = _default_hf_token_path()
    token_source = None
    token_for_hub: str | bool | None = None
    if env_token:
        token_source = "environment (`HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`)"
        token_for_hub = env_token
    elif token_path.exists() and token_path.read_text().strip():
        token_source = f"token file at {token_path}"
        token_for_hub = True

    try:
        from huggingface_hub import HfApi
    except Exception:
        if token_source is not None:
            return CheckResult(
                name="Hugging Face auth",
                ok=False,
                summary=f"found {token_source}, but `huggingface_hub` is not installed so gated-model access could not be verified",
                fix=(
                    "Install the Hub client in this environment and log in again.\n"
                    "python -m pip install -U huggingface_hub\n"
                    "hf auth login"
                ),
            )
        return CheckResult(
            name="Hugging Face auth",
            ok=False,
            summary="no Hugging Face login detected",
            fix=(
                "Install the Hub CLI and log in with an approved token for MahmoodLab/UNI2-h.\n"
                "python -m pip install -U huggingface_hub\n"
                "hf auth login"
            ),
        )

    api = HfApi()
    try:
        api.model_info("MahmoodLab/UNI2-h", token=token_for_hub)
        if token_source is not None:
            summary = f"verified access to MahmoodLab/UNI2-h using {token_source}"
        else:
            summary = "verified access to MahmoodLab/UNI2-h using the active Hugging Face login"
        return CheckResult(name="Hugging Face auth", ok=True, summary=summary)
    except Exception as exc:
        hf_cli = shutil.which("hf")
        cli_hint = None
        if hf_cli is not None:
            ok, output = _run_command([hf_cli, "auth", "whoami"])
            if ok:
                cli_hint = output
        summary = f"could not verify access to MahmoodLab/UNI2-h: {exc}"
        if cli_hint:
            summary += f" (current CLI login: {cli_hint})"
        return CheckResult(
            name="Hugging Face auth",
            ok=False,
            summary=summary,
            fix=(
                "Log in with a token that has approved access to MahmoodLab/UNI2-h.\n"
                "python -m pip install -U huggingface_hub\n"
                "hf auth login"
            ),
        )


def check_artifacts(*, model_dir: str | Path | None, preset_name: str = "s4_new_multiclass") -> CheckResult:
    try:
        bundle = resolve_model_bundle(preset_name=preset_name, model_dir=model_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        return CheckResult(
            name="QC model artifacts",
            ok=False,
            summary=str(exc),
            fix="Keep the bundled files in `models/qc` or pass `--model-dir /path/to/model_dir`.",
        )
    summary = f"found required model artifacts under {bundle['model_dir']}"
    if bundle["manifest_path"] is not None:
        summary += f" (manifest verified: {bundle['manifest_path'].name})"
    return CheckResult(
        name="QC model artifacts",
        ok=True,
        summary=summary,
    )


def run_doctor(*, trident_dir: str | Path, model_dir: str | Path | None) -> int:
    checks = [
        check_python_runtime(),
        check_openslide(),
        check_vips(),
        check_trident(trident_dir),
        check_hugging_face_auth(),
        check_artifacts(model_dir=model_dir),
    ]
    print("he-quality doctor")
    print(f"python: {sys.executable}")
    print("")
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.summary}")
        if check.fix:
            print(check.fix)
        print("")
    if all(check.ok for check in checks):
        print("All required gates passed.")
        print("You can now run:")
        print("he-quality run-qc --input-path /path/to/wsi_or_folder --output-dir /path/to/output")
        return 0
    print("One or more required gates failed. Fix the items above and rerun `he-quality doctor`.")
    return 1
