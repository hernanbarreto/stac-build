import torch
import torchvision.transforms as T
import cv2
import numpy as np
import gc
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DinoEvaluator:
    def __init__(self):
        self.model = None
        import os, sys
        os.environ["XFORMERS_DISABLED"] = "1"
        to_remove = [k for k in sys.modules if "dinov2" in k]
        for k in to_remove:
            del sys.modules[k]
        self.device = torch.device("cpu")
        self.patch_size = 14
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(
                (518, 518), 
                interpolation=T.InterpolationMode.BICUBIC, 
                antialias=True
            ),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def load_model(self):
        if self.model is None:
            logger.info("[DINOv2] Loading dinov2_vits14 on CPU...")
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model.to(self.device)
            self.model.eval()
            logger.info("[DINOv2] Model loaded.")

    def unload_model(self):
        if self.model is not None:
            logger.info("[DINOv2] Unloading model from CPU...")
            del self.model
            self.model = None
            gc.collect()

    @torch.no_grad()
    def evaluate_mask_presence(self, frames_dir: Path, ref_frame_idx: int, ref_mask: np.ndarray, threshold: float = 0.5):
        """
        Evaluate which frames contain the object indicated by ref_mask in ref_frame_idx.
        Returns a list of boolean values, one for each frame.
        """
        if self.model is None:
            self.load_model()
            
        frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
        num_frames = len(frame_files)
        if num_frames == 0:
            return []
            
        if ref_frame_idx >= num_frames:
            logger.error(f"[DINOv2] Invalid ref_frame_idx {ref_frame_idx}")
            return [True] * num_frames
            
        # 1. Process reference frame
        ref_img_path = frame_files[ref_frame_idx]
        ref_img = cv2.imread(str(ref_img_path))
        if ref_img is None:
            return [True] * num_frames
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = ref_img.shape[:2]
        
        # Ensure mask is boolean and same shape as image
        if ref_mask.shape[:2] != (orig_h, orig_w):
            ref_mask = cv2.resize(ref_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        ref_mask_bool = ref_mask > 0
        
        if not np.any(ref_mask_bool):
            logger.warning("[DINOv2] Reference mask is empty, returning all frames")
            return [True] * num_frames
            
        ref_tensor = self.transform(ref_img).unsqueeze(0).to(self.device)
        # Check if the hub model structure requires careful feature extraction
        try:
            ref_features_dict = self.model.forward_features(ref_tensor)
            ref_features = ref_features_dict["x_norm_patchtokens"] # [1, H_p * W_p, C]
        except Exception:
            # Fallback if structure changed
            ref_features = self.model(ref_tensor, is_training=False) # might return tuple or dict
            if isinstance(ref_features, tuple):
                ref_features = ref_features[0]
        
        # Resize mask to patch grid size (37x37 for 518/14)
        hp, wp = 518 // self.patch_size, 518 // self.patch_size
        mask_resized = cv2.resize(ref_mask.astype(np.uint8), (wp, hp), interpolation=cv2.INTER_NEAREST)
        mask_flat = mask_resized.flatten() > 0
        
        if not np.any(mask_flat):
            # Fallback if resizing destroyed small mask
            mask_flat[len(mask_flat)//2] = True
            
        # Extract object embedding (average of mask patches)
        ref_features_flat = ref_features[0] # [N, C]
        obj_embedding = ref_features_flat[mask_flat].mean(dim=0, keepdim=True) # [1, C]
        obj_embedding = torch.nn.functional.normalize(obj_embedding, dim=1)
        
        # 2. Process all frames
        results = []
        for i, f_path in enumerate(frame_files):
            if i == ref_frame_idx:
                results.append(True)
                continue
                
            img = cv2.imread(str(f_path))
            if img is None:
                results.append(False)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            features_dict = self.model.forward_features(tensor)
            features = features_dict["x_norm_patchtokens"][0] # [N, C]
            features = torch.nn.functional.normalize(features, dim=1)
            
            # Compute cosine similarity
            sim = torch.mm(features, obj_embedding.T) # [N, 1]
            max_sim = sim.max().item()
            
            results.append(max_sim >= threshold)
            
            if i % 10 == 0:
                logger.info(f"[DINOv2] Evaluated {i}/{num_frames} frames (max_sim={max_sim:.3f})")
                
        return results

# Singleton
_evaluator = DinoEvaluator()

def get_dino_evaluator() -> DinoEvaluator:
    return _evaluator
