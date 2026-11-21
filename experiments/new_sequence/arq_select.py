"""THE ARCHITECTURE OF 2026-11-21, step 2 — the keyframes: overlap of 92 to 96 % with the last chosen keyframe.

The selection of v2_select.py (the walk of the video from its first frame, each frame compared with the last chosen
keyframe on AMB3R's poses and depth) with the band the user set for this architecture (2026-11-21: "entonces elegimos de
92 a 96"): above 96 % the same view again, 92 – 96 % a candidate, under 92 % the best candidate is chosen.

    python arq_select.py <scan dir> <amb3r_video.npz> <out.json>"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import v2_select as S  # noqa: E402

S.OVERLAP_REPEAT = 0.96
S.OVERLAP_HOLE = 0.92
S.BATCH = 8                # frames compared with the last keyframe at once: a keyframe every few frames at this band

if __name__ == "__main__":
    sys.exit(S.main())
