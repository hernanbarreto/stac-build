# STAC-BUILD: Segmentation Manager
# Manages unified segments.json file for all segmented objects
# Supports multiple object types, multiple instances per type, per-chunk indices
# Implements batch SAM3 processing with ID propagation across batches

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from threading import Lock
from dataclasses import dataclass, field
import gc
import torch


# Predefined colors for object types (will cycle through)
from config import cfg

# Get colors from config (Formerly Hardcoded)
SEGMENT_COLORS = cfg["visualization"]["segment_colors"]


@dataclass
class SegmentInstance:
    """A single instance of an object (e.g., one specific trash can)."""
    instance_id: int
    global_id: str  # Unique identifier like "trash_can_1"
    chunks: Dict[int, List[int]] = field(default_factory=dict)  # chunk_id -> point_indices
    obb: Optional[Dict] = None  # Oriented Bounding Box
    
    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "global_id": self.global_id,
            "chunks": {str(k): v for k, v in self.chunks.items()},
            "obb": self.obb
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SegmentInstance':
        return cls(
            instance_id=data["instance_id"],
            global_id=data["global_id"],
            chunks={int(k): v for k, v in data.get("chunks", {}).items()},
            obb=data.get("obb")
        )


@dataclass
class ObjectType:
    """A type of object (e.g., 'trash can', 'bucket')."""
    type_id: int
    label: str
    color: str
    instances: List[SegmentInstance] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "type_id": self.type_id,
            "label": self.label,
            "color": self.color,
            "instances": [inst.to_dict() for inst in self.instances]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ObjectType':
        return cls(
            type_id=data["type_id"],
            label=data["label"],
            color=data.get("color", "#FFFFFF"),
            instances=[SegmentInstance.from_dict(i) for i in data.get("instances", [])]
        )


class SegmentationManager:
    """
    Manages the unified segments.json file.
    Handles adding new object types, instances, and chunk data.
    Thread-safe for concurrent access.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.segments_file = self.output_dir / "segments.json"
        self.lock = Lock()
        self.object_types: Dict[str, ObjectType] = {}  # label -> ObjectType
        self.next_type_id = 1
        self._load()
    
    def _load(self):
        """Load existing segments.json if it exists."""
        if self.segments_file.exists():
            try:
                with open(self.segments_file, 'r') as f:
                    data = json.load(f)
                
                for ot_data in data.get("object_types", []):
                    ot = ObjectType.from_dict(ot_data)
                    self.object_types[ot.label] = ot
                    self.next_type_id = max(self.next_type_id, ot.type_id + 1)
                
                print(f"[SegmentationManager] Loaded {len(self.object_types)} object types from {self.segments_file}")
            except Exception as e:
                print(f"[SegmentationManager] Error loading segments.json: {e}")
                self.object_types = {}
    
    def _save(self):
        """Save current state to segments.json."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "object_types": [ot.to_dict() for ot in self.object_types.values()]
        }
        
        with open(self.segments_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_segment(self,
                    label: str,
                    chunk_id: int,
                    point_indices: List[int],
                    instance_id: int = 1,
                    obb: Optional[Dict] = None,
                    replace: bool = False) -> str:
        """
        Add or update a segment for an object.

        Args:
            label: Object type label (e.g., "white trash can")
            chunk_id: Which chunk this segment belongs to
            point_indices: List of point indices in the chunk's PLY
            instance_id: Which instance of this object type (from SAM3)
            obb: Optional oriented bounding box
            replace: If True, replace existing indices for this chunk instead of merging

        Returns:
            global_id of the instance
        """
        with self.lock:
            # Get or create object type
            if label not in self.object_types:
                color = SEGMENT_COLORS[(self.next_type_id - 1) % len(SEGMENT_COLORS)]
                self.object_types[label] = ObjectType(
                    type_id=self.next_type_id,
                    label=label,
                    color=color,
                    instances=[]
                )
                self.next_type_id += 1
                print(f"[SegmentationManager] Created new object type: '{label}' (id={self.object_types[label].type_id})")

            obj_type = self.object_types[label]

            # Find or create instance
            instance = None
            for inst in obj_type.instances:
                if inst.instance_id == instance_id:
                    instance = inst
                    break

            if instance is None:
                # Create new instance
                global_id = f"{label.replace(' ', '_')}_{instance_id}"
                instance = SegmentInstance(
                    instance_id=instance_id,
                    global_id=global_id,
                    chunks={},
                    obb=None
                )
                obj_type.instances.append(instance)
                print(f"[SegmentationManager] Created new instance: '{global_id}'")

            # Add/update chunk data
            if replace:
                # Replace existing indices completely
                instance.chunks[chunk_id] = sorted(point_indices)
            else:
                # Merge with existing indices (avoid duplicates)
                existing_indices = instance.chunks.get(chunk_id, [])
                merged = list(set(existing_indices + point_indices))
                instance.chunks[chunk_id] = sorted(merged)
            
            # Update OBB if provided
            if obb is not None:
                instance.obb = obb
            
            # Save to disk (skip if inside a batch operation)
            if not getattr(self, '_skip_save', False):
                self._save()
            
            return instance.global_id
    
    def get_all_segments(self) -> Dict:
        """Get full segments data for viewer."""
        with self.lock:
            return {
                "version": "1.0",
                "object_types": [ot.to_dict() for ot in self.object_types.values()]
            }
    
    def clear(self):
        """Clear all segments (for new session)."""
        with self.lock:
            self.object_types = {}
            self.next_type_id = 1
            if self.segments_file.exists():
                self.segments_file.unlink()
            print("[SegmentationManager] Cleared all segments")
    
    def compute_obb_for_instance(self, label: str, instance_id: int, 
                                  all_points: Dict[int, np.ndarray]) -> Optional[Dict]:
        """
        Compute OBB for an instance across all its chunks.
        
        Args:
            label: Object type label
            instance_id: Instance ID
            all_points: Dict of chunk_id -> point_cloud array
            
        Returns:
            OBB dict or None
        """
        with self.lock:
            if label not in self.object_types:
                return None
            
            obj_type = self.object_types[label]
            instance = None
            for inst in obj_type.instances:
                if inst.instance_id == instance_id:
                    instance = inst
                    break
            
            if instance is None:
                return None
            
            # Collect all 3D points for this instance
            all_segment_points = []
            for chunk_id, indices in instance.chunks.items():
                if chunk_id in all_points:
                    pc = all_points[chunk_id]
                    valid_indices = [i for i in indices if i < len(pc)]
                    if valid_indices:
                        all_segment_points.append(pc[valid_indices, :3])
            
            if not all_segment_points:
                return None
            
            points = np.concatenate(all_segment_points, axis=0)
            
            # Simple AABB for now (TODO: proper OBB with PCA)
            min_pt = np.min(points, axis=0)
            max_pt = np.max(points, axis=0)
            center = (min_pt + max_pt) / 2
            extents = max_pt - min_pt
            
            obb = {
                "center": center.tolist(),
                "extents": extents.tolist(),
                "rotation": [[1,0,0], [0,1,0], [0,0,1]]  # Identity for AABB
            }
            
            instance.obb = obb
            self._save()
            
            return obb

    def rebuild_from_chunks(self):
        """
        Scan output directory for chunk_XXX_segments.json files
        and rebuild the unified segments.json.
        """
        print(f"[SegmentationManager] Rebuilding from chunks in {self.output_dir}...")

        # Clear current state (memory only, file will be overwritten)
        self.object_types = {}
        self.next_type_id = 1

        # Find all chunk segment files
        segment_files = sorted(self.output_dir.glob("chunk_*_segments.json"))

        if not segment_files:
            print("[SegmentationManager] No chunk segment files found.")
            return

        self._skip_save = True
        try:
            for sf in segment_files:
                print(f"[SegmentationManager] Processing {sf.name}...")
                try:
                    with open(sf, 'r') as f:
                        data = json.load(f)

                    chunk_id = data.get("chunk_id")
                    if chunk_id is None:
                        print(f"[SegmentationManager] Warning: No chunk_id in {sf.name}")
                        continue

                    objects = data.get("objects", [])
                    print(f"  Found {len(objects)} objects in chunk {chunk_id}")

                    for obj in objects:
                        label = obj.get("label", "Unknown")
                        instance_id = obj.get("id", 1)
                        point_indices_data = obj.get("point_indices", [])

                        actual_indices = []
                        if isinstance(point_indices_data, list):
                            if len(point_indices_data) > 0 and isinstance(point_indices_data[0], dict):
                                for item in point_indices_data:
                                    if "indices" in item:
                                        actual_indices.extend(item["indices"])
                            else:
                                actual_indices = point_indices_data

                        if actual_indices:
                            self.add_segment(
                                label=label,
                                chunk_id=chunk_id,
                                point_indices=actual_indices,
                                instance_id=instance_id,
                                replace=True  # Replace when rebuilding from chunks
                            )
                except Exception as e:
                    print(f"[SegmentationManager] Error processing {sf.name}: {e}")
        finally:
            self._skip_save = False

        print(f"[SegmentationManager] Rebuild complete: {len(self.object_types)} object types found.")
        self._save()


@dataclass
class TrackedObject:
    """Tracks an object across SAM3 batches for ID propagation."""
    global_id: int
    label: str
    last_mask: Optional[np.ndarray] = None
    last_frame: int = -1
    last_centroid: Optional[Tuple[float, float]] = None


class BatchSegmentationProcessor:
    """
    Processes SAM3 segmentation in batches with object ID propagation.
    Ensures consistent object IDs across the entire session.
    """

    def __init__(self, sam3_wrapper):
        self.sam3 = sam3_wrapper
        self.tracked_objects: Dict[str, List[TrackedObject]] = {}  # prompt -> [TrackedObject]
        self.next_global_id = 1
        self.iou_threshold = 0.3  # Minimum IoU to consider same object

    def reset(self):
        """Reset tracking state for a new session."""
        self.tracked_objects = {}
        self.next_global_id = 1

    def process_session(self, frames_dir: Path, prompt: str,
                       batch_size: int = None,
                       keyframe_interval: int = None) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Process entire session in batches with ID propagation.

        Args:
            frames_dir: Directory containing all session frames
            prompt: Text prompt for segmentation
            batch_size: Frames per batch (from config if None)
            keyframe_interval: Interval for extracting masks (from config if None)

        Returns:
            Dict[frame_global, Dict[object_id, binary_mask]]
        """
        # Get config values
        if batch_size is None:
            batch_size = cfg["mapanything"]["chunk_size"]
        if keyframe_interval is None:
            keyframe_interval = cfg["models"]["segmentation"]["keyframe_interval"]

        # Discover all frames
        frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
        n_frames = len(frame_files)

        if n_frames == 0:
            print(f"[BatchSeg] No frames found in {frames_dir}")
            return {}

        print(f"[BatchSeg] Processing {n_frames} frames in batches of {batch_size}")
        print(f"[BatchSeg] Prompt: '{prompt}', keyframe_interval: {keyframe_interval}")

        # Initialize tracking for this prompt
        if prompt not in self.tracked_objects:
            self.tracked_objects[prompt] = []

        all_masks: Dict[int, Dict[int, np.ndarray]] = {}  # frame_global -> {obj_id: mask}

        # Process in batches
        for batch_start in range(0, n_frames, batch_size):
            batch_end = min(batch_start + batch_size, n_frames)
            batch_frames = frame_files[batch_start:batch_end]

            print(f"[BatchSeg] Processing batch {batch_start//batch_size + 1}: frames {batch_start}-{batch_end-1}")

            # Run SAM3 on this batch
            batch_masks = self._process_batch(batch_frames, prompt, keyframe_interval)

            # Process results with ID propagation
            for local_frame_idx, frame_masks in batch_masks.items():
                frame_global = batch_start + local_frame_idx

                if frame_global not in all_masks:
                    all_masks[frame_global] = {}

                if "out_binary_masks" not in frame_masks:
                    continue

                binary_masks = frame_masks["out_binary_masks"]
                if hasattr(binary_masks, "cpu"):
                    binary_masks = binary_masks.cpu().numpy()

                if binary_masks.ndim == 2:
                    binary_masks = binary_masks[None, ...]

                # Match each detection to tracked objects
                for det_idx in range(binary_masks.shape[0]):
                    det_mask = binary_masks[det_idx]
                    obj_id = self._match_or_create_object(prompt, det_mask, frame_global)
                    all_masks[frame_global][obj_id] = det_mask

        print(f"[BatchSeg] Processed {len(all_masks)} frames with masks")
        return all_masks

    def _process_batch(self, frame_paths: List[Path], prompt: str,
                      keyframe_interval: int) -> Dict[int, Any]:
        """Process a single batch of frames."""
        import tempfile
        import shutil

        # Create temp directory with batch frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for i, fp in enumerate(frame_paths):
                dst = temp_path / f"{i:05d}.jpg"
                shutil.copy2(fp, dst)

            # Run SAM3
            return self.sam3.process_chunk(str(temp_path), prompt, keyframe_interval)

    def _match_or_create_object(self, prompt: str, mask: np.ndarray, frame_idx: int) -> int:
        """
        Match a detection to an existing tracked object or create a new one.
        Uses IoU overlap and centroid distance for matching.
        """
        tracked = self.tracked_objects.get(prompt, [])

        # Compute mask centroid
        mask_bool = mask > 0
        if not np.any(mask_bool):
            # Empty mask - create new object anyway
            return self._create_new_object(prompt, mask, frame_idx)

        y_coords, x_coords = np.where(mask_bool)
        centroid = (float(np.mean(y_coords)), float(np.mean(x_coords)))

        best_match = None
        best_iou = 0.0

        for obj in tracked:
            if obj.last_mask is None:
                continue

            # Compute IoU
            iou = self._compute_iou(mask, obj.last_mask)

            if iou > best_iou and iou > self.iou_threshold:
                best_iou = iou
                best_match = obj

        if best_match is not None:
            # Update tracked object
            best_match.last_mask = mask
            best_match.last_frame = frame_idx
            best_match.last_centroid = centroid
            return best_match.global_id
        else:
            # Create new object
            return self._create_new_object(prompt, mask, frame_idx, centroid)

    def _create_new_object(self, prompt: str, mask: np.ndarray,
                          frame_idx: int, centroid: Tuple[float, float] = None) -> int:
        """Create a new tracked object."""
        obj_id = self.next_global_id
        self.next_global_id += 1

        obj = TrackedObject(
            global_id=obj_id,
            label=prompt,
            last_mask=mask,
            last_frame=frame_idx,
            last_centroid=centroid
        )

        if prompt not in self.tracked_objects:
            self.tracked_objects[prompt] = []

        self.tracked_objects[prompt].append(obj)
        print(f"[BatchSeg] Created new object ID {obj_id} for '{prompt}' at frame {frame_idx}")
        return obj_id

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union between two masks."""
        # Handle size mismatch
        if mask1.shape != mask2.shape:
            import cv2
            mask2 = cv2.resize(mask2.astype(np.float32),
                              (mask1.shape[1], mask1.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        m1 = mask1 > 0
        m2 = mask2 > 0

        intersection = np.sum(m1 & m2)
        union = np.sum(m1 | m2)

        if union == 0:
            return 0.0

        return float(intersection) / float(union)

    def unload_resources(self):
        """Unload SAM3 and free memory."""
        if self.sam3 is not None:
            self.sam3.unload_model()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[BatchSeg] Resources unloaded")
