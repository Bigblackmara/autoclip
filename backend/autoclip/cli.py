"""AutoClip command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__, config, paths, system

app = typer.Typer(
    name="autoclip",
    help="Turn long video into caption-burned 9:16 clips — locally.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

OK = "[green]OK[/green]"
WARN = "[yellow]WARN[/yellow]"
FAIL = "[red]FAIL[/red]"


def _status(passed: bool, warn_only: bool = False) -> str:
    if passed:
        return OK
    return WARN if warn_only else FAIL


def _ffmpeg_install_hint() -> str:
    """Platform-specific install advice that actually yields a usable ffmpeg.

    The distinction matters more than it looks: the default package on both
    Windows and macOS is a reduced build without libass, so following the
    obvious instruction produces an ffmpeg that installs fine and then cannot
    burn a single caption.
    """
    if sys.platform == "darwin":
        return (
            "  macOS: [cyan]brew install ffmpeg-full[/cyan] — not [cyan]ffmpeg[/cyan], "
            "which Homebrew now ships as a reduced build with no libass.\n"
            "  If you already installed the plain formula:\n"
            "    [cyan]brew install ffmpeg-full && brew unlink ffmpeg && "
            "brew link --force --overwrite ffmpeg-full[/cyan]"
        )
    if sys.platform == "win32":
        return (
            "  Windows: [cyan]winget install Gyan.FFmpeg[/cyan] — the [bold]full[/bold] "
            "build, not [cyan]Gyan.FFmpeg.Essentials[/cyan], which omits libass."
        )
    return (
        "  Debian/Ubuntu: [cyan]sudo apt install ffmpeg[/cyan]\n"
        "  Fedora: [cyan]sudo dnf install ffmpeg[/cyan] (RPM Fusion)\n"
        "  Arch: [cyan]sudo pacman -S ffmpeg[/cyan]"
    )


@app.command()
def version() -> None:
    """Print the AutoClip version."""
    console.print(f"autoclip {__version__}")


@app.command()
def doctor() -> None:
    """Check that this machine can run the pipeline, and explain anything missing."""
    report = system.refresh()
    settings = config.load()

    console.print()
    console.print(
        Panel.fit(
            Text.from_markup(
                f"[bold]AutoClip {__version__}[/bold]\n{report.platform}\nHome: {paths.root()}"
            ),
            border_style="cyan",
        )
    )

    remediation: list[str] = []

    # --- Runtime -----------------------------------------------------------
    table = Table(title="Runtime", show_header=True, header_style="bold", title_justify="left")
    table.add_column("Check")
    table.add_column("Status", width=6)
    table.add_column("Detail")

    table.add_row("Python", _status(report.python_ok), report.python_version)
    if not report.python_ok:
        remediation.append(
            "Python must be >=3.11,<3.13 — MediaPipe publishes no wheels for 3.13+, "
            "so the reframe stage cannot run. Create the environment with "
            "[cyan]uv venv --python 3.11[/cyan]."
        )

    ff = report.ffmpeg
    table.add_row("ffmpeg", _status(ff.found), ff.version or "not found")
    table.add_row("ffprobe", _status(ff.ffprobe_found), ff.path or "not found")
    if not ff.found or not ff.ffprobe_found:
        remediation.append(
            "Install ffmpeg and make sure both [cyan]ffmpeg[/cyan] and [cyan]ffprobe[/cyan] "
            f"are on PATH.\n{_ffmpeg_install_hint()}"
        )
    else:
        table.add_row("  libass (captions)", _status(ff.has_libass), "subtitle burn-in")
        table.add_row("  fontconfig", _status(ff.has_fontconfig, warn_only=True), "font resolution")
        table.add_row("  libx264", _status(ff.has_libx264), "software H.264 encode")
        if ff.has_nvenc and not ff.nvenc_works:
            table.add_row("  h264_nvenc", WARN, ff.nvenc_error or "listed but not usable")
            remediation.append(
                "This ffmpeg build lists [cyan]h264_nvenc[/cyan] but the GPU driver is "
                "too old for the NVENC API it was built against, so exports will use "
                "the CPU encoder. Update your NVIDIA driver to enable GPU encoding — "
                "or ignore this, since software encoding produces identical quality, "
                "just slower."
            )
        else:
            table.add_row(
                "  h264_nvenc",
                _status(ff.nvenc_works, warn_only=True),
                "GPU encode" if ff.nvenc_works else "not available (software encode will be used)",
            )
        missing = ff.missing_filters
        table.add_row(
            "  required filters",
            _status(not missing),
            "all present" if not missing else f"missing: {', '.join(missing)}",
        )
        optional_missing = [f for f in system.OPTIONAL_FILTERS if f not in ff.filters]
        if optional_missing:
            table.add_row(
                "  optional filters",
                WARN,
                f"missing: {', '.join(optional_missing)}",
            )
        if not ff.has_libass or "ass" in missing:
            remediation.append(
                "This ffmpeg build has no libass, so captions cannot be burned in — "
                f"which is most of what AutoClip does.\n{_ffmpeg_install_hint()}"
            )

    console.print()
    console.print(table)

    # --- Acceleration ------------------------------------------------------
    gpu = report.gpu
    accel_table = Table(
        title="Acceleration", show_header=True, header_style="bold", title_justify="left"
    )
    accel_table.add_column("Check")
    accel_table.add_column("Status", width=6)
    accel_table.add_column("Detail")

    accel_table.add_row("Mode", OK, gpu.accel.upper())
    if gpu.name:
        vram = f", {gpu.vram_mb} MiB VRAM" if gpu.vram_mb else ""
        cc = f", compute {gpu.compute_capability}" if gpu.compute_capability else ""
        accel_table.add_row("Device", OK, f"{gpu.name}{vram}{cc}")
    if gpu.driver_version:
        accel_table.add_row("Driver", OK, gpu.driver_version)

    if gpu.name and not gpu.ctranslate2_cuda:
        if gpu.cuda_libraries_missing:
            missing = " and ".join(gpu.cuda_libraries_missing)
            detail = f"GPU found, but {missing} cannot be loaded"
        else:
            missing = "the CUDA runtime"
            detail = "CTranslate2 cannot see the GPU"
        accel_table.add_row("CUDA for Whisper", FAIL, detail)
        remediation.append(
            f"An NVIDIA GPU is present but Whisper can't use it — {missing} is missing. "
            "Install the CUDA runtime libraries with "
            "[cyan]uv pip install 'autoclip[gpu]'[/cyan], then re-run doctor. "
            "Transcription runs on the CPU (same result, slower) until this is fixed."
        )
    elif gpu.ctranslate2_cuda:
        accel_table.add_row("CUDA for Whisper", OK, "GPU usable (cuBLAS and cuDNN load)")

    if gpu.supported_compute_types:
        accel_table.add_row(
            "Supported compute types", OK, ", ".join(sorted(gpu.supported_compute_types))
        )

    reason = ""
    if gpu.accel == "cuda" and not gpu.compute_type.startswith("float16"):
        reason = " (this GPU has no usable fp16 path)"
    accel_table.add_row("Whisper compute type", OK, f"{gpu.compute_type}{reason}")

    override = settings.whisper.compute_type
    if override:
        usable = not gpu.supported_compute_types or override in gpu.supported_compute_types
        accel_table.add_row(
            "  override in config",
            _status(usable, warn_only=True),
            f"forced to {override}" if usable else f"{override} is unsupported — ignored",
        )
        if not usable:
            remediation.append(
                f"[cyan]whisper.compute_type[/cyan] is set to [cyan]{override}[/cyan], which "
                "this device does not support, so it is being ignored. Clear it in "
                f"{paths.config_path()} to silence this."
            )

    console.print()
    console.print(accel_table)

    # --- Python dependencies ----------------------------------------------
    deps = report.deps
    dep_table = Table(
        title="Dependencies", show_header=True, header_style="bold", title_justify="left"
    )
    dep_table.add_column("Package")
    dep_table.add_column("Status", width=6)
    dep_table.add_column("Used for")

    dep_table.add_row("faster-whisper", _status(deps.faster_whisper), "transcription")
    dep_table.add_row("mediapipe", _status(deps.mediapipe), "face detection / reframe")
    dep_table.add_row("scenedetect", _status(deps.scenedetect), "shot boundaries")
    dep_table.add_row(
        "whisperx",
        _status(deps.whisperx, warn_only=True),
        "speaker diarization" + ("" if deps.whisperx else " (optional extra)"),
    )

    if not deps.whisperx and settings.whisper.diarization:
        remediation.append(
            "Diarization is enabled in settings but WhisperX isn't installed. "
            "Run [cyan]uv pip install 'autoclip[diarization]'[/cyan] and set a HuggingFace "
            "token with [cyan]autoclip config set-secret huggingface_token[/cyan]."
        )

    for missing, extra in (
        (not deps.faster_whisper, "faster-whisper"),
        (not deps.mediapipe, "mediapipe"),
        (not deps.scenedetect, "scenedetect"),
    ):
        if missing:
            remediation.append(
                f"[cyan]{extra}[/cyan] is not installed — reinstall AutoClip's core "
                "dependencies with [cyan]uv pip install -e '.[dev]'[/cyan]."
            )

    console.print()
    console.print(dep_table)

    # --- Providers ---------------------------------------------------------
    prov_table = Table(
        title="LLM providers", show_header=True, header_style="bold", title_justify="left"
    )
    prov_table.add_column("Provider")
    prov_table.add_column("Status", width=6)
    prov_table.add_column("Detail")

    for name in config.KEYED_PROVIDERS:
        has_key = config.get_secret(name, settings) is not None
        model = settings.provider(name).model or "no model set"
        active = " [cyan](active)[/cyan]" if settings.active_provider == name else ""
        prov_table.add_row(
            f"{name}{active}",
            _status(has_key, warn_only=True),
            f"{model}" if has_key else "no API key stored",
        )

    ollama_active = " [cyan](active)[/cyan]" if settings.active_provider == "ollama" else ""
    if deps.ollama_running:
        model_list = ", ".join(deps.ollama_models[:4]) or "no models pulled"
        prov_table.add_row(f"ollama{ollama_active}", OK, model_list)
    else:
        prov_table.add_row(
            f"ollama{ollama_active}", WARN, "not running on localhost:11434 (optional)"
        )

    if not any(config.get_secret(n, settings) for n in config.KEYED_PROVIDERS) and not (
        deps.ollama_running
    ):
        remediation.append(
            "No LLM provider is usable yet. Add an API key with "
            "[cyan]autoclip config set-secret anthropic[/cyan] (or openai / gemini), "
            "or install Ollama for a fully local setup."
        )

    console.print()
    console.print(prov_table)

    # --- Storage -----------------------------------------------------------
    store_table = Table(
        title="Storage", show_header=True, header_style="bold", title_justify="left"
    )
    store_table.add_column("Check")
    store_table.add_column("Status", width=6)
    store_table.add_column("Detail")

    store_table.add_row(
        "Secret storage",
        _status(deps.keyring_backend, warn_only=True),
        "OS keyring" if deps.keyring_backend else "no keyring backend — plaintext fallback",
    )
    if not deps.keyring_backend:
        remediation.append(
            "No OS keyring backend is available, so API keys would be written to "
            f"{paths.config_path()} in plaintext. On headless Linux install "
            "[cyan]keyrings.alt[/cyan] or a Secret Service provider."
        )
    if settings.insecure_secret_storage:
        store_table.add_row("Stored secrets", WARN, "one or more secrets are in plaintext")

    store_table.add_row("Home", OK, str(paths.root()))

    console.print()
    console.print(store_table)

    # --- Verdict -----------------------------------------------------------
    console.print()
    if report.ready:
        console.print(
            Panel.fit(
                "[bold green]Ready.[/bold green] The core pipeline can run on this machine.",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold red]Not ready.[/bold red] Resolve the items below, then re-run "
                "[cyan]autoclip doctor[/cyan].",
                border_style="red",
            )
        )

    if remediation:
        console.print()
        console.print("[bold]What to do:[/bold]")
        for i, item in enumerate(remediation, 1):
            console.print(f"  [bold]{i}.[/bold] {item}")
        console.print()

    raise typer.Exit(0 if report.ready else 1)


config_app = typer.Typer(help="Inspect and modify AutoClip settings.", no_args_is_help=True)
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show() -> None:
    """Print current settings (secrets are never displayed)."""
    settings = config.load()
    console.print_json(settings.model_dump_json(indent=2))


@config_app.command("path")
def config_path_cmd() -> None:
    """Print the path to config.json."""
    console.print(str(paths.config_path()))


@config_app.command("set-secret")
def config_set_secret(
    key: str = typer.Argument(
        ...,
        help="Provider name (anthropic, openai, gemini) or 'huggingface_token'.",
    ),
) -> None:
    """Store an API key or token. The value is prompted for, never passed as an argument.

    Prompting rather than accepting a flag keeps the secret out of your shell
    history and out of the process list.
    """
    valid = (*config.KEYED_PROVIDERS, config.HF_TOKEN_KEY)
    if key not in valid:
        console.print(f"[red]Unknown secret '{key}'.[/red] Expected one of: {', '.join(valid)}")
        raise typer.Exit(2)

    value = typer.prompt(f"Value for {key}", hide_input=True).strip()
    if not value:
        console.print("[yellow]Empty value — nothing stored.[/yellow]")
        raise typer.Exit(1)

    paths.ensure_layout()
    secure = config.set_secret(key, value)
    if secure:
        console.print(f"[green]Stored {key} in the OS keyring.[/green]")
    else:
        console.print(
            f"[yellow]No keyring backend available — {key} was written to "
            f"{paths.config_path()} in plaintext.[/yellow]"
        )


@config_app.command("delete-secret")
def config_delete_secret(key: str = typer.Argument(..., help="Secret to remove.")) -> None:
    """Remove a stored secret."""
    config.delete_secret(key)
    console.print(f"[green]Removed {key}.[/green]")


@app.command()
def init() -> None:
    """Create the AutoClip home directory and initialise the database."""
    from . import db

    version_applied = db.init()
    console.print(f"[green]Initialised[/green] {paths.root()} (schema v{version_applied})")


@app.command()
def clip(
    target: str = typer.Argument(..., help="A YouTube URL, or a path to a local media file."),
    provider: str = typer.Option("", "--provider", "-p", help="Override the active provider."),
    model: str = typer.Option("", "--whisper-model", help="Override the Whisper model."),
    max_clips: int = typer.Option(0, "--max-clips", "-n", help="Override the clip count."),
    style: str = typer.Option("", "--style", "-s", help="Caption style preset."),
    ratio: str = typer.Option("", "--ratio", "-r", help="Output ratio: 9:16, 1:1, or 16:9."),
    diarize: bool = typer.Option(
        False, "--diarize/--no-diarize", help="Label speakers (needs the diarization extra)."
    ),
    centre_crop: bool = typer.Option(
        False, "--centre-crop", help="Skip face tracking and centre-crop everything."
    ),
) -> None:
    """Turn a video into captioned vertical clips."""
    import asyncio

    from . import db
    from .db import store
    from .db.models import Job, new_id
    from .pipeline import ingest, runner

    db.init()
    settings = config.load()

    if provider:
        settings.active_provider = provider  # type: ignore[assignment]
    if model:
        settings.whisper.model = model
    if max_clips:
        settings.clips.max_clips = max_clips
    if style:
        settings.export.caption_style = style
    if ratio:
        settings.export.ratio = ratio  # type: ignore[assignment]
    if diarize:
        settings.whisper.diarization = True
    if centre_crop:
        settings.export.__dict__["centre_crop"] = True

    # --- ingest ---------------------------------------------------------
    try:
        with console.status("[cyan]Fetching source...", spinner="dots"):
            if ingest.is_youtube_url(target):
                source = ingest.ingest_youtube(target, settings.ingest)
            else:
                source = ingest.ingest_file(Path(target))
    except ingest.IngestError as exc:
        console.print(f"\n[red]Ingest failed.[/red] {exc}")
        raise typer.Exit(1) from exc

    store.create_source(source)
    console.print(
        f"[green]Source:[/green] {source.title} "
        f"({_format_duration(source.duration_s)}, {source.width}x{source.height})"
    )

    job = store.create_job(
        Job(
            id=new_id(),
            source_id=source.id,
            provider=settings.active_provider,
            settings=settings.model_dump(mode="json"),
        )
    )

    # --- run ------------------------------------------------------------
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Starting", total=1.0)

        def on_progress(event: runner.ProgressEvent) -> None:
            progress.update(task, completed=event.overall, description=event.message)

        try:
            clips = asyncio.run(
                runner.PipelineRunner(job, source, settings=settings, on_progress=on_progress).run()
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            raise typer.Exit(130) from None
        except Exception as exc:
            progress.stop()
            console.print(f"\n[red]Pipeline failed.[/red] {exc}")
            raise typer.Exit(1) from exc

    # --- report ---------------------------------------------------------
    table = Table(title=f"{len(clips)} clips", header_style="bold", title_justify="left")
    table.add_column("#", width=3)
    table.add_column("Score", width=5)
    table.add_column("Length", width=7)
    table.add_column("Title")

    for c in clips:
        table.add_row(str(c.rank), str(c.score), f"{c.duration_s:.0f}s", c.title or "(untitled)")

    console.print()
    console.print(table)
    console.print(f"\n[green]Exported to[/green] {paths.exports_dir() / job.id}")


@app.command()
def jobs(limit: int = typer.Option(15, "--limit", "-n")) -> None:
    """List recent jobs."""
    from . import db
    from .db import store

    db.init()
    table = Table(title="Recent jobs", header_style="bold", title_justify="left")
    table.add_column("ID", width=16)
    table.add_column("Status", width=10)
    table.add_column("Stage", width=12)
    table.add_column("Progress", width=8)
    table.add_column("Source")

    for job in store.list_jobs(limit=limit):
        source = store.get_source(job.source_id)
        colour = {
            "done": "green",
            "failed": "red",
            "running": "cyan",
            "cancelled": "yellow",
        }.get(job.status, "white")
        table.add_row(
            job.id,
            f"[{colour}]{job.status}[/{colour}]",
            job.current_stage,
            f"{job.progress * 100:.0f}%",
            (source.title if source else "?")[:50],
        )

    console.print(table)


@app.command()
def providers() -> None:
    """Check which LLM providers are reachable right now."""
    import asyncio

    from .providers import PROVIDERS, build_provider

    settings = config.load()

    async def check_all():
        results = []
        for name in PROVIDERS:
            try:
                provider = build_provider(name, settings)
                results.append(await provider.health_check())
            except Exception as exc:
                from .providers import ProviderStatus

                results.append(ProviderStatus(name=name, available=False, detail=str(exc)[:120]))
        return results

    with console.status("[cyan]Checking providers...", spinner="dots"):
        statuses = asyncio.run(check_all())

    table = Table(title="LLM providers", header_style="bold", title_justify="left")
    table.add_column("Provider", width=12)
    table.add_column("Status", width=6)
    table.add_column("Detail")

    for status in statuses:
        active = " [cyan]*[/cyan]" if settings.active_provider == status.name else ""
        table.add_row(
            f"{status.name}{active}", _status(status.available, warn_only=True), status.detail
        )

    console.print(table)
    if any(s.available for s in statuses):
        console.print("\n[dim]* = active provider[/dim]")


@app.command("fetch-models")
def fetch_models() -> None:
    """Download the ML model bundles the reframe stage needs."""
    from . import models

    for key, spec in models.MODELS.items():
        if models.is_available(key):
            console.print(f"[green]OK[/green] {spec.filename} already downloaded")
            continue
        with console.status(f"[cyan]Downloading {spec.filename}...", spinner="dots"):
            try:
                path = models.ensure(key)
            except models.ModelDownloadError as exc:
                console.print(f"[red]FAILED[/red] {spec.filename}\n{exc}")
                raise typer.Exit(1) from exc
        console.print(f"[green]OK[/green] {path}")


@app.command("update-ytdlp")
def update_ytdlp() -> None:
    """Update yt-dlp, which YouTube changes force often."""
    import shutil
    import subprocess
    import sys

    console.print("[cyan]Updating yt-dlp...[/cyan]")
    # The `default` extra carries yt-dlp-ejs, the solver YouTube's player
    # challenge needs. Environments made by uv have no pip, so use uv there.
    package = "yt-dlp[default]"
    has_pip = (
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"], capture_output=True, check=False
        ).returncode
        == 0
    )
    uv = shutil.which("uv")
    if has_pip:
        command = [sys.executable, "-m", "pip", "install", "--upgrade", package]
    elif uv:
        command = [uv, "pip", "install", "--python", sys.executable, "--upgrade", package]
    else:
        console.print(
            "[red]Update failed.[/red] This environment has neither pip nor uv. "
            f"Install uv, or run: {sys.executable} -m ensurepip"
        )
        raise typer.Exit(1)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        console.print(f"[red]Update failed.[/red]\n{result.stderr}")
        raise typer.Exit(1)

    import importlib.metadata

    try:
        version_installed = importlib.metadata.version("yt-dlp")
        console.print(f"[green]yt-dlp is now at {version_installed}.[/green]")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        console.print("[green]Update complete.[/green]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open a browser once the server is up."
    ),
) -> None:
    """Start the AutoClip web app."""
    import threading
    import webbrowser

    import uvicorn

    from . import db
    from .app import static_dir

    db.init()

    if static_dir() is None:
        console.print(
            "[yellow]The frontend has not been built.[/yellow] The API will still work "
            "at /docs. To build the UI:\n"
            "  [cyan]cd frontend && npm install && npm run build[/cyan]\n"
        )

    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    console.print(f"[green]AutoClip[/green] starting on [cyan]{url}[/cyan]")

    if host == "0.0.0.0":  # noqa: S104
        console.print(
            "[yellow]Binding to 0.0.0.0 exposes AutoClip to your whole network.[/yellow] "
            "There is no authentication — only do this on a network you trust."
        )

    if open_browser and not reload:
        # Delayed so the browser doesn't race the server's first bind.
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "autoclip.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command()
def styles() -> None:
    """List the available caption styles."""
    from .pipeline.captions import PRESETS

    table = Table(title="Caption styles", header_style="bold", title_justify="left")
    table.add_column("Key", width=14)
    table.add_column("Name", width=14)
    table.add_column("Description")

    for style in PRESETS.values():
        table.add_row(style.key, style.label, style.description)

    console.print(table)


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# ---------------------------------------------------------------------------
# live — the Live Watch command group.
#
# The CLI is a read-and-configure surface, never a manager host: starting
# capture involves watcher tasks, subprocesses, and an asyncio loop, and a
# forgotten terminal session must never be the thing keeping a 24/7 watch
# alive. `autoclip serve` owns the manager lifecycle; these commands only
# touch the persistent records (the registry + the store + config), which
# `serve` picks up on its next start/reconnect.
# ---------------------------------------------------------------------------

live_app = typer.Typer(invoke_without_command=True)
app.add_typer(live_app, name="live")


@live_app.callback()
def live(ctx: typer.Context) -> None:
    """Watch live streams and clip their viral moments. Run with `autoclip serve`."""
    if ctx.invoked_subcommand is None:
        _live_status()


#: Stream states under which a capture subprocess is alive right now;
#: everything else is pending (waiting_for_live), user-paused (off), or
#: terminal. The vocabulary lives in live/states.py and is the one thing
#: the manager, API, and this CLI all agree on.
_LIVE_WATCHING_STATES = frozenset(("connecting", "watching", "clipping", "reconnecting"))

#: Rich colour per stream state — the same mapping the streams panel (F1)
#: uses for its state pills, so the terminal and the web read identically.
_LIVE_STATE_COLOURS = {
    "watching": "green",
    "clipping": "yellow",
    "reconnecting": "magenta",
    "error": "red",
    "off": "bright_black",
}

#: Clip-row statuses, for the `live clips` colouring. Same vocabulary the
#: store's live_clips.status column carries (detecting → ready/rejected).
_LIVE_CLIP_COLOURS = {
    "ready": "green",
    "detecting": "cyan",
    "queued": "cyan",
    "rendering": "cyan",
    "failed": "red",
    "rejected": "bright_black",
}


def _live_settings() -> Any:
    """LiveSettings as persisted in config.json's ``live`` section.

    ``Settings`` deliberately has no ``live`` field yet (the section is
    additive), so this reads the raw file — :func:`config.load` drops keys
    the model doesn't know about. A missing, unreadable, or half-written
    section degrades to the frozen defaults, exactly the contract
    ``live/manager._load_settings`` applies on the serve side.
    """
    import json

    from pydantic import ValidationError

    from .live.settings import LiveSettings

    raw: Any = None
    path = paths.config_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data.get("live") if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            raw = None
    if raw is None:
        return LiveSettings()
    try:
        return LiveSettings.model_validate(raw)
    except ValidationError:
        console.print(
            "[yellow]The live section of config.json is invalid — showing defaults.[/yellow]"
        )
        return LiveSettings()


def _live_status() -> None:
    """The default view: every registered stream, its state, and the caps."""
    from . import db
    from .live import registry
    from .live import store as live_store

    paths.ensure_layout()
    db.init()

    streams: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Registration order — the registry is the watch list `serve` loads
    # watchers from — then any DB rows it doesn't know, so a stream added
    # while the server was down still shows up.
    try:
        registered = registry.list_streams()
    except Exception:
        registered = []  # a registry read failure must not hide the DB view
    for cfg in registered:
        seen.add(cfg.id)
        row = live_store.get_stream(cfg.id)
        streams.append(
            {
                "id": cfg.id,
                "label": cfg.label,
                "url": cfg.url,
                "enabled": cfg.enabled,
                "state": row.state if row else ("off" if not cfg.enabled else "waiting_for_live"),
                "detail": row.state_detail if row else "",
            }
        )
    for row in live_store.list_streams():
        if row.id not in seen:
            streams.append(
                {
                    "id": row.id,
                    "label": row.label,
                    "url": row.url,
                    "enabled": row.enabled,
                    "state": row.state,
                    "detail": row.state_detail,
                }
            )

    table = Table(
        title="Live Watch streams", show_header=True, header_style="bold", title_justify="left"
    )
    table.add_column("ID", width=16)
    table.add_column("Label", width=18)
    table.add_column("State", width=18)
    table.add_column("URL")

    watching = 0
    for stream in streams:
        state = stream["state"]
        if state in _LIVE_WATCHING_STATES:
            watching += 1
        colour = _LIVE_STATE_COLOURS.get(state, "cyan")
        cell = f"[{colour}]{state}[/{colour}]"
        if not stream["enabled"] and state != "off":
            cell += " [dim](paused)[/dim]"
        table.add_row(stream["id"], stream["label"] or "—", cell, stream["url"])

    console.print(table)

    settings = _live_settings()
    feature = "[green]on[/green]" if settings.enabled else "[red]off[/red]"
    console.print(
        f"[green]{watching}[/green] stream(s) actively watched · "
        f"{len(streams)} registered · feature {feature}"
    )
    console.print(
        f"[dim]caps: {settings.workers.max_concurrent_streams} concurrent streams · "
        f"{settings.workers.clip_queue_workers} render worker(s) · "
        f"clip queue {settings.workers.clip_queue_max}[/dim]"
    )


def _live_find(stream_id: str) -> tuple[Any, Any]:
    """One stream's (registry entry, store row) — either side may be None."""
    from .live import registry
    from .live import store as live_store

    return registry.get(stream_id), live_store.get_stream(stream_id)


def _live_require(stream_id: str) -> tuple[Any, Any]:
    """Both views of an existing stream, or exit with the shared not-found error."""
    cfg, row = _live_find(stream_id)
    if cfg is None and row is None:
        console.print(f"[red]No Live Watch stream with id '{stream_id}'.[/red]")
        raise typer.Exit(1)
    return cfg, row


@live_app.command("status")
def live_status() -> None:
    """Print the Live Watch status table (the default view for bare `autoclip live`)."""
    _live_status()


@live_app.command("add")
def live_add(
    url: str = typer.Argument(..., help="Stream URL to watch (Twitch, YouTube Live, Kick...)."),
    label: str = typer.Option("", "--label", "-l", help="Human-readable name for the stream."),
) -> None:
    """Register a stream on the watch list; `autoclip serve` starts watching it.

    Registration is pure configuration — the registry entry the server
    loads watchers from, plus the store row that will carry its runtime
    state. Actually starting a watcher here would tie a 24/7 watch to the
    lifetime of one terminal session, so the CLI deliberately leaves that
    to `serve`.
    """
    from . import db
    from .live import registry
    from .live import store as live_store
    from .live.models import LiveStreamRow
    from .live.settings import LiveStreamConfig

    paths.ensure_layout()
    db.init()

    url = url.strip()
    if not url:
        console.print("[red]The stream URL must not be empty.[/red]")
        raise typer.Exit(2)

    try:
        # An empty id asks the registry to mint one — it owns id creation.
        cfg = registry.upsert(LiveStreamConfig(id="", url=url, label=label))
    except ValueError as exc:
        console.print(f"[red]Could not register the stream.[/red] {exc}")
        raise typer.Exit(1) from exc

    live_store.upsert_stream(
        LiveStreamRow(id=cfg.id, url=cfg.url, label=cfg.label, enabled=cfg.enabled)
    )

    console.print(f"[green]Added.[/green] {cfg.url}")
    if cfg.label:
        console.print(f"  label: {cfg.label}")
    console.print(f"  id: {cfg.id}")
    console.print("[dim]`autoclip serve` starts watching it on its next start or reconnect.[/dim]")


@live_app.command("remove")
def live_remove(
    stream_id: str = typer.Argument(..., help="Stream id, as printed by `autoclip live`."),
) -> None:
    """Remove a stream: its registry entry, its DB row, and every clip of it go."""
    from . import db
    from .live import registry
    from .live import store as live_store

    paths.ensure_layout()
    db.init()

    _live_require(stream_id)
    registry.delete(stream_id)
    live_store.delete_stream(stream_id)

    console.print(f"[green]Removed.[/green] {stream_id}")


@live_app.command("pause")
def live_pause(
    stream_id: str = typer.Argument(..., help="Stream id, as printed by `autoclip live`."),
) -> None:
    """Stop watching a stream: capture stops, its configuration is kept."""
    from . import db
    from .live import registry
    from .live import store as live_store
    from .live.states import StreamState

    paths.ensure_layout()
    db.init()

    _live_require(stream_id)
    registry.set_enabled(stream_id, False)
    live_store.update_stream_state(stream_id, state=StreamState.OFF.value)

    console.print(f"[green]Paused.[/green] {stream_id}")
    console.print(f"[dim]resume with [cyan]autoclip live resume {stream_id}[/cyan][/dim]")


@live_app.command("resume")
def live_resume(
    stream_id: str = typer.Argument(..., help="Stream id, as printed by `autoclip live`."),
) -> None:
    """Resume a paused stream; `serve` launches a fresh watcher for it."""
    from . import db
    from .live import registry
    from .live import store as live_store
    from .live.states import StreamState

    paths.ensure_layout()
    db.init()

    _live_require(stream_id)
    registry.set_enabled(stream_id, True)
    # Resume is a fresh watcher start in the manager's model — it launches
    # from waiting_for_live, mirroring live/manager.resume_stream.
    live_store.update_stream_state(stream_id, state=StreamState.WAITING_FOR_LIVE.value)

    console.print(f"[green]Resumed.[/green] {stream_id}")


@live_app.command("clips")
def live_clips(
    stream_id: str = typer.Option("", "--stream", "-s", help="Only clips from this stream."),
    limit: int = typer.Option(20, "--limit", "-n", help="Show at most this many clips."),
) -> None:
    """List the latest live clips, newest first."""
    from . import db
    from .live import store as live_store

    paths.ensure_layout()
    db.init()

    rows = live_store.list_clips(stream_id=stream_id or None, limit=max(1, limit))

    table = Table(title="Live clips", header_style="bold", title_justify="left")
    table.add_column("ID", width=16)
    table.add_column("Status", width=10)
    table.add_column("Score", width=5, justify="right")
    table.add_column("Length", width=6, justify="right")
    table.add_column("Title")

    for c in rows:
        colour = _LIVE_CLIP_COLOURS.get(c.status, "white")
        duration = max(0.0, c.clip_end_s - c.clip_start_s)
        table.add_row(
            c.id,
            f"[{colour}]{c.status}[/{colour}]",
            str(c.score),
            f"{duration:.0f}s",
            c.title or "(untitled)",
        )

    console.print(table)
    if not rows:
        console.print("[dim]No live clips yet — they appear as watched streams produce them.[/dim]")


@live_app.command("settings")
def live_settings() -> None:
    """Print the live settings section from config.json."""
    from . import db

    paths.ensure_layout()
    db.init()

    settings = _live_settings()
    console.print_json(settings.model_dump_json(indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
