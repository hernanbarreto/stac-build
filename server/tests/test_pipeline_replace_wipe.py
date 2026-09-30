"""Replace mode leaves the session with its two inputs only: the frame images and
the original video (USER 2026-09-28: "no debe dejar nada más que los frames y el
video original ... absolutamente nada más")."""

from __future__ import annotations


def test_replace_leaves_only_frames_and_video(tmp_path):
    from pipeline_manager import PipelineManager

    s = tmp_path / "sess"
    (s / "frames").mkdir(parents=True)
    (s / "frames" / "000000.jpg").write_bytes(b"jpg")
    (s / "frames" / "000001.png").write_bytes(b"png")
    (s / "frames" / "quality_features.json").write_text("{}")     # intake I0
    (s / "frames" / "witness_frames.json").write_text("{}")       # intake I1
    (s / "frames" / "sub").mkdir()
    (s / "frames" / "sub" / "a.txt").write_text("")
    (s / "source_video.mp4").write_bytes(b"video")
    for d in ("output/potree", "output/precision", "intake/da3_focal", "frames_valid",
              "user_prefs", ".output.wiping-42"):
        (s / d).mkdir(parents=True)
        (s / d / "f").write_text("")
    (s / "output" / "corrections.jsonl").write_text('{"a": 1}\n')
    (s / "output" / "geometry_epoch.json").write_text('{"epoch": 3}')
    (s / "stray.txt").write_text("")

    PipelineManager._wipe_outputs_for_replace(s, s / "output")

    left = sorted(str(p.relative_to(s)) for p in s.rglob("*"))
    # output/ is recreated EMPTY for the run; nothing else survives
    assert left == ["frames", "frames/000000.jpg", "frames/000001.png", "output",
                    "source_video.mp4"], left
    assert (s / "frames" / "000000.jpg").read_bytes() == b"jpg"
    assert (s / "source_video.mp4").read_bytes() == b"video"


def test_replace_on_a_bare_session_is_a_no_op(tmp_path):
    from pipeline_manager import PipelineManager

    s = tmp_path / "sess"
    (s / "frames").mkdir(parents=True)
    (s / "frames" / "000000.jpg").write_bytes(b"jpg")
    (s / "source_video.mov").write_bytes(b"video")
    PipelineManager._wipe_outputs_for_replace(s, s / "output")
    left = sorted(str(p.relative_to(s)) for p in s.rglob("*"))
    assert left == ["frames", "frames/000000.jpg", "output", "source_video.mov"], left


def test_a_wiped_run_never_deletes_what_its_own_stages_produced():
    """pccr 2026-09-30: after the full wipe of "Reconstruir", the per-stage cleanup
    deleted cleaned_cloud.ply — published by the reconstruction stage (F0-F7 +
    f7_cloud) — the moment the cloud stage started."""
    import inspect
    import pipeline_manager as PM
    src = inspect.getsource(PM)
    i = src.index("wiped_this_run = False")
    assert "wiped_this_run = True" in src[i:i + 200]
    j = src.index("self._cleanup_stage_outputs(\n", i)
    guard = src[src.rindex("if ", i, j):j]
    assert "not wiped_this_run" in guard, guard
