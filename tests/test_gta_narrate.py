"""Static caption box + narrator voiceover (gta/narrate.py)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from autoclip.config import ExportSettings
from autoclip.gta import narrate
from autoclip.gta.qa import measure_loudness, track_lengths
from autoclip.pipeline import ffmpeg


def _clip(path: Path, seconds: float = 20.0) -> Path:
    subprocess.run(
        [
            ffmpeg.ffmpeg_path(), "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=s=1080x1920:r=30:d={seconds}",
            "-f", "lavfi", "-i", f"sine=f=330:d={seconds}:sample_rate=48000",
            "-ac", "2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )
    return path


def _fake_speak(text: str, out: Path, *, voice: str, speed: float) -> float:
    import soundfile as sf

    length = 0.25 * len(text.split())
    t = np.arange(int(24000 * length)) / 24000
    sf.write(str(out), (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 24000)
    return length


class TestPlan:
    def test_outro_only_on_long_enough_clips(self) -> None:
        assert narrate.plan_voice(10.0, 3.0, 2.0) == (narrate.INTRO_AT_S, None)
        intro_at, outro_at = narrate.plan_voice(30.0, 3.0, 2.0)
        assert intro_at == narrate.INTRO_AT_S
        assert outro_at == pytest.approx(30.0 - narrate.OUTRO_END_PAD_S - 2.0)

    def test_intro_waits_for_the_flash_forward(self) -> None:
        assert narrate.plan_voice(30.0, 3.0, 0.0, lead_s=2.0)[0] == pytest.approx(2.15)

    def test_outro_dropped_when_it_would_crowd_the_intro(self) -> None:
        assert narrate.plan_voice(15.0, 6.0, 4.0)[1] is None

    def test_duck_expression(self) -> None:
        assert narrate.duck_expression([], -11.0) == "1"
        expr = narrate.duck_expression([(0.15, 3.0)], -11.0)
        assert expr.startswith("1-0.71")


class TestLines:
    def test_fallback_uses_the_caption_sentence(self) -> None:
        n = narrate.fallback_lines(
            "Kai Cenat", "Kai meets a shark in GTA 6. The panic is unreal. #gta6", "Shark"
        )
        assert n.intro == "Kai meets a shark in GTA 6."
        assert n.outro in narrate.OUTROS

    def test_ai_lines_cleaned(self) -> None:
        async def complete(system, user, config):
            return (
                '{"intro": "Speed finds a baby in GTA 6 \U0001F602 and panics",'
                ' "outro": "What would you do"}'
            )

        n, source = narrate.write_lines(
            streamer="Speed", hook="Baby", caption="", summary="", transcript="", complete=complete
        )
        assert source == "ai"
        assert n.intro == "Speed finds a baby in GTA 6 and panics."
        assert n.outro == "What would you do?"

    def test_ai_failure_falls_back(self) -> None:
        async def complete(system, user, config):
            raise RuntimeError("quota")

        n, source = narrate.write_lines(
            streamer="Speed", hook="Baby", caption="Speed finds a baby.", summary="",
            transcript="", complete=complete,
        )
        assert (source, n.intro) == ("rules", "Speed finds a baby.")


class TestBox:
    def test_box_wraps_evenly_and_fits(self, tmp_path: Path) -> None:
        w, h = narrate.caption_png("The Bank Roasted His Credit", tmp_path / "b.png")
        assert w <= narrate.BOX_MAX_W + 8
        long_w, long_h = narrate.caption_png(
            "Kai Cenat gets proposed to by a random GTA 6 NPC and completely loses it",
            tmp_path / "c.png",
        )
        assert long_h > h and long_w <= narrate.BOX_MAX_W + 8

    def test_box_sits_at_the_bottom_above_the_platform_bar(self) -> None:
        assert narrate.box_top(("bottom", 1590), 200) == 1390
        assert narrate.box_top(("bottom", 1900), 200) == 1920 - 200 - narrate.BOX_MIN_BOTTOM
        assert narrate.box_top(("centre", 640), 200) == 540
        assert narrate.box_top(("above", 200), 200) == narrate.BOX_MIN_TOP


class TestApply:
    def test_caption_and_voice_keep_sync_and_loudness(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clip = _clip(tmp_path / "clip.mp4")
        monkeypatch.setattr(narrate, "speak", _fake_speak)
        out = narrate.apply(
            clip,
            caption_text="He Robbed The Wrong Bank",
            narration=narrate.Narration("Speed walks into a bank with a plan.", "Would you do it?"),
            anchor=("above", 656),
            work_dir=tmp_path / "work",
            voice="af_heart",
            speed=1.1,
            duck_db=-11.0,
            outro=True,
            lead_s=0.0,
            loudness_lufs=-14.0,
            true_peak_db=-1.0,
            export_settings=ExportSettings(prefer_hardware_encoder=False),
        )
        assert out["edits"] == ["caption box", "voiceover"]
        assert len(out["voiceover"]) == 2
        video_s, audio_s = track_lengths(clip)
        assert abs(video_s - 20.0) < 0.1 and abs(video_s - audio_s) < 0.15
        lufs, _ = measure_loudness(clip)
        assert lufs is not None and abs(lufs + 14.0) < 1.0

    def test_finish_never_raises(self, tmp_path: Path) -> None:
        from autoclip.gta.settings import GtaSettings

        out = narrate.finish(
            tmp_path / "missing.mp4",
            settings=GtaSettings(),
            export_settings=ExportSettings(),
            anchor=("above", 656),
            lead_s=0.0,
            work_dir=tmp_path / "w",
            hook="Hook",
            narration=None,
        )
        assert out["edits"] == [] and "error" in out
