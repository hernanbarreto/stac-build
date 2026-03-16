# STAC-BUILD: Frames Package
# Re-exports for backward compatibility
from frames.storage import get_frame_storage, FrameStorage
from frames.quality import analyze_frames, save_manifest
from frames.selector import select_keyframes, load_selected_frames
