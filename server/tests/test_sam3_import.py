#!/usr/bin/env python3
"""Quick smoke test: verify SAM 3.1 model_builder is importable."""
import sys
sys.path.insert(0, "../vendor/sam3")

try:
    from sam3.model_builder import build_sam3_predictor
    print("build_sam3_predictor imported OK")
    
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    print("build_sam3_multiplex_video_predictor imported OK")
    
    from sam3.model_builder import download_ckpt_from_hf
    print("download_ckpt_from_hf imported OK")
    
    print("\nSAM 3.1 vendor update: SUCCESS")
except Exception as e:
    print(f"IMPORT FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
