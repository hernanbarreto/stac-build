# STAC-BUILD: Segmentation Manager
# Manages unified segments.json file for all segmented objects
# Supports multiple object types, multiple instances per type, per-chunk indices

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from threading import Lock
from dataclasses import dataclass, field


# Predefined colors for object types (will cycle through)
SEGMENT_COLORS = [
    "#FFFF00",  # Yellow
    "#00FF00",  # Green
    "#FF00FF",  # Magenta
    "#00FFFF",  # Cyan
    "#FF8000",  # Orange
    "#8000FF",  # Purple
    "#FF0080",  # Pink
    "#0080FF",  # Blue
]


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
                    obb: Optional[Dict] = None) -> str:
        """
        Add or update a segment for an object.
        
        Args:
            label: Object type label (e.g., "white trash can")
            chunk_id: Which chunk this segment belongs to
            point_indices: List of point indices in the chunk's PLY
            instance_id: Which instance of this object type (from SAM3)
            obb: Optional oriented bounding box
            
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
            existing_indices = instance.chunks.get(chunk_id, [])
            # Merge indices (avoid duplicates)
            merged = list(set(existing_indices + point_indices))
            instance.chunks[chunk_id] = sorted(merged)
            
            # Update OBB if provided
            if obb is not None:
                instance.obb = obb
            
            # Save to disk
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
