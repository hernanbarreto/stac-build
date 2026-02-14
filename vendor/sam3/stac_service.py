# stac_service.py
# Backend de Inteligencia Artificial para STAC - VERSIÓN FINAL FUNCIONAL
# Ejecutar en entorno conda: conda activate sam3

import sys
import os
import json
import shutil
import traceback
import torch
import numpy as np
from PIL import Image
from huggingface_hub import login

# --- CONFIGURACIÓN ESTÁTICA ---
OUTPUT_BASE_DIR = os.path.join(os.getcwd(), "scene")
MASKS_DIR = os.path.join(OUTPUT_BASE_DIR, "output", "masks")
JSON_DIR = os.path.join(OUTPUT_BASE_DIR, "output", "json")
INPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "input")

# Autenticación Silenciosa
try:
    login(token="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")
except:
    pass

# GPU Cleanup at startup
def cleanup_gpu():
    """Clear any zombie CUDA processes at startup."""
    import gc
    import subprocess
    
    # Try nvidia-smi reset
    try:
        subprocess.run(['nvidia-smi', '--gpu-reset', '-i', '0'], 
                      capture_output=True, timeout=10)
    except:
        pass
    
    # PyTorch cleanup
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except:
        pass
    
    gc.collect()

# Run cleanup at import
cleanup_gpu()
print("🔄 GPU cleanup complete", file=sys.stderr)

# --- UTILIDADES MATEMÁTICAS (NMS) ---
def compute_containment(mask_big, mask_small):
    """Calcula porcentaje de inclusión de mask_small dentro de mask_big"""
    intersection = np.logical_and(mask_big, mask_small).sum()
    area_small = mask_small.sum()
    if area_small == 0: return 0.0
    return intersection / area_small

def apply_nms_hierarchical(candidates, containment_threshold=0.75):
    """Aplica Non-Maximum Suppression (NMS) basado en el área y score."""
    for c in candidates: c['area'] = np.sum(c['mask_bool'])
    candidates = sorted(candidates, key=lambda x: (x['area'], x['score']), reverse=True)
    final_list = []
    
    while candidates:
        parent = candidates.pop(0)
        final_list.append(parent)
        remaining = []
        for child in candidates:
            containment = compute_containment(parent['mask_bool'], child['mask_bool'])
            if containment < containment_threshold:
                remaining.append(child)
        candidates = remaining
    return final_list

# --- CLASE PRINCIPAL DEL SERVICIO ---
class STACService:
    def __init__(self):
        self.log("⏳ Iniciando STAC Service (Modo Unificado)...")
        self.video_predictor = None
        self.device = "cpu"
        
        self.current_image_np = None
        self.current_image_pil = None
        self.video_session_id = None
        self.temp_video_dir = None
        
        self._init_directories()
        self._load_models()

    def log(self, msg):
        print(json.dumps({"type": "log", "message": msg}), flush=True)

    def _init_directories(self):
        #self.log("🧹 Limpiando espacio de trabajo...")
        #dirs = [MASKS_DIR, JSON_DIR]
        #for d in dirs:
        #    if os.path.exists(d):
        #        shutil.rmtree(d)
        #    os.makedirs(d, exist_ok=True)
        #os.makedirs(INPUT_DIR, exist_ok=True)
        pass

    def _load_models(self):
        try:
            from sam3.model_builder import build_sam3_video_predictor
            
            if torch.cuda.is_available():
                self.device = "cuda"
                self.log(f"✅ GPU Detectada: {torch.cuda.get_device_name(0)}")
            else:
                self.log("⚠️ Ejecutando en CPU (Lento)")

            # Video Predictor para TODO (Texto y Puntos)
            self.log("📦 Cargando SAM3 Video Predictor (Unificado)...")
            self.video_predictor = build_sam3_video_predictor()
            
            self.log("✅ Modelo SAM3 Unificado cargado correctamente.")
            
        except Exception as e:
            self.log(f"❌ Error Fatal cargando SAM3: {e}")
            self.log(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)

    # --- COMANDOS DE IMAGEN Y TEXTO ---

    def load_image(self, path):
        if not os.path.exists(path):
            return {"status": "error", "message": f"Imagen no encontrada: {path}"}
        
        try:
            self.current_image_pil = Image.open(path).convert("RGB")
            self.current_image_np = np.array(self.current_image_pil)
            
            # Preparar "video" de 1 frame para refinamiento
            self.temp_video_dir = os.path.join(INPUT_DIR, "temp_video_frames")
            if os.path.exists(self.temp_video_dir):
                shutil.rmtree(self.temp_video_dir)
            os.makedirs(self.temp_video_dir, exist_ok=True)
            
            # Guardar como frame 00000.jpg (formato requerido por SAM3)
            frame_path = os.path.join(self.temp_video_dir, "00000.jpg")
            self.current_image_pil.save(frame_path, quality=95)
            
            # Iniciar sesión de video
            try:
                response = self.video_predictor.handle_request({
                    "type": "start_session",
                    "resource_path": self.temp_video_dir
                })
                self.video_session_id = response.get("session_id")
                self.log(f"✅ Sesión de video iniciada: {self.video_session_id}")
            except Exception as e:
                self.log(f"⚠️ No se pudo iniciar sesión de video: {e}")
                self.video_session_id = None
            
            w, h = self.current_image_pil.size
            self.log(f"📸 Imagen cargada: {w}x{h}px")
            return {"status": "success", "width": w, "height": h}
            
        except Exception as e:
            self.log(f"Error cargando imagen: {e}")
            self.log(f"Traceback: {traceback.format_exc()}")
            return {"status": "error", "message": str(e)}

    def segment_text(self, concepts, category_types=None, category_priors=None):
        self.log(f"👉 Solicitud segment_text recibida. Conceptos: {str(concepts)}")
        
        # Default empty dict if not provided
        if category_types is None:
            category_types = {}
        if category_priors is None:
            category_priors = {}
        
        # LIMPIAR OUTPUT antes de generar nuevas máscaras
        self.log("🧹 Limpiando carpetas de salida...")
        for d in [MASKS_DIR, JSON_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        
        if self.video_session_id is None:
            self.log("❌ Error: segment_text llamado sin video_session_id activo")
            return {"status": "error", "message": "No hay sesión activa. Carga una imagen primero."}

        all_candidates = []
        global_id = 0

        self.log(f"🔍 Buscando {len(concepts)} conceptos con Video Predictor...")

        for concept in concepts:
            try:
                # Detectar con texto en el video predictor frame 0
                response = self.video_predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": self.video_session_id,
                    "frame_index": 0,
                    "text": concept
                })
                
                if "outputs" in response:
                    outputs = response["outputs"]
                    # La salida es un diccionario con arrays numpy: {'out_obj_ids': [...], 'out_probs': [...], 'out_binary_masks': [...]}
                    if "out_obj_ids" in outputs and "out_binary_masks" in outputs:
                        out_obj_ids = outputs["out_obj_ids"]
                        out_probs = outputs.get("out_probs", [])
                        out_masks = outputs["out_binary_masks"]
                        
                        num_obj = len(out_obj_ids)
                        for i in range(num_obj):
                            obj_id = int(out_obj_ids[i])
                            score = float(out_probs[i]) if i < len(out_probs) else 1.0
                            mask_np = out_masks[i]
                            
                            # Filtro básico de score
                            if score < 0.30: continue

                            # Asegurar máscara bool 2D
                            if mask_np.ndim > 2: mask_np = mask_np.squeeze()
                            mask_bool = mask_np > 0

                            # Bounding Box
                            y_idxs, x_idxs = np.where(mask_bool)
                            if len(x_idxs) > 0:
                                x1, x2 = np.min(x_idxs), np.max(x_idxs)
                                y1, y2 = np.min(y_idxs), np.max(y_idxs)
                                bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                            else:
                                bbox = [0,0,0,0]

                            all_candidates.append({
                                "label": concept,
                                "score": score,
                                "mask_bool": mask_bool,
                                "bbox": bbox,
                                "global_id": global_id,
                                "video_obj_id": obj_id # Guardamos el ID real del video
                            })
                            global_id += 1

            except Exception as e:
                self.log(f"⚠️ Error detectando '{concept}': {e}")

        # Filtrado Inteligente (NMS Jerárquico)
        self.log(f"🧠 Filtrando {len(all_candidates)} candidatos crudos...")
        final_objects = apply_nms_hierarchical(all_candidates, containment_threshold=0.8)

        # Generación de Artefactos (Recortes PNG con transparencia)
        results = []
        final_id = 0
        self.detected_objects = {} # Reiniciar mapeo

        for obj in final_objects:
            # Crear imagen RGBA (con canal alpha para transparencia)
            h, w = obj['mask_bool'].shape
            cutout_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Copiar RGB donde hay máscara
            cutout_rgba[obj['mask_bool'], :3] = self.current_image_np[obj['mask_bool']]
            # Alpha = 255 donde hay máscara, 0 donde no
            cutout_rgba[obj['mask_bool'], 3] = 255
            
            filename = f"mask_{final_id}.png"
            Image.fromarray(cutout_rgba, mode='RGBA').save(os.path.join(MASKS_DIR, filename))
            
            # Usar directamente el ID real del video para evitar confusiones
            real_id = obj.get("video_obj_id", final_id)
            
            # Get type from category_types if provided
            cat_type = category_types.get(obj['label'], 'scene')
            cat_prior = category_priors.get(obj['label'], None)

            # Save JSON metadata
            json_filename = f"mask_{final_id}.json"
            metadata = {
                "id": real_id,
                "name": obj['label'],
                "type": cat_type,
                "y0": True,  # Default: object base goes to Y=0 (ground level)
                "prior": cat_prior,  # Size priors [min, max] for height/width/length
                "score": obj['score'],
                "bbox": obj['bbox']
            }
            with open(os.path.join(MASKS_DIR, json_filename), 'w') as f:
                json.dump(metadata, f, indent=2)

            results.append({
                "id": final_id,         # UI ID (Unique, 0, 1, 2...)
                "video_id": real_id,    # Backend ID (SAM ID: 5, 23...)
                "label": obj['label'],
                "score": obj['score'],
                "mask_file": filename,
                "type": cat_type,
                "y0": True,  # Default: ground level
                "bbox": obj['bbox']
            })
            final_id += 1

        self.log(f"✅ Segmentación completada: {len(results)} objetos únicos.")
        return {"status": "success", "detections": results}
    
    # --- COMANDO CRÍTICO: REFINAR RIEL (Puntos Interactivos) ---
    def refine_rail(self, points, labels, rail_type, obj_id=None):
        self.log(f"👉 Solicitud refine_rail recibida. Type: {rail_type}, Points: {len(points)}, ObjID: {obj_id}")
        
        if self.video_session_id is None:
            self.log("❌ Error: refine_rail llamado sin video_session_id")
            return {"status": "error", "message": "No hay sesión de video disponible. Recarga la imagen."}
            
        try:
            # DEBUG: Validate Image Dimensions vs Point Coordinates
            h, w = self.current_image_np.shape[:2]
            self.log(f"🔍 DEBUG DIMENSIONS: Internal Image Shape: {w}x{h}")
            
            # CRITICAL FIX RE-VERIFIED: The model expects NORMALIZED coordinates [0-1] by default
            # because 'rel_coordinates=True' is the default in add_prompt.
            # We must normalize using the INTERNAL image dimensions (though we use the original ones for safety)
            # Actually, we should use the dimensions of the image we found the points on (the full res one).
            
            norm_points = []
            for p in points:
                px, py = p
                # Normalize to 0-1 range
                nx = px / w
                ny = py / h
                norm_points.append([nx, ny])
                self.log(f"   Norm Pt: ({px:.1f}, {py:.1f}) -> ({nx:.4f}, {ny:.4f})")

            # Añadir prompt de puntos en el frame 0
            
            # CRITICAL FIX: Use a NEW UNIQUE ID for the rail to separate it from the original object
            # We want to segment the RAIL, not update the TRAIN mask.
            # ID 10001 for Left, 10002 for Right.
            rail_id_map = {"left": 10001, "right": 10002}
            target_obj_id = rail_id_map.get(rail_type, 20000)
            
            self.log(f"   Refinando Riel: '{rail_type}'. Usando ID dedicado: {target_obj_id} (Ignorando ID original {obj_id} para crear nueva máscara)")

            response = self.video_predictor.handle_request({
                "type": "add_prompt",
                "session_id": self.video_session_id,
                "frame_index": 0,
                "obj_id": target_obj_id, # Usamos el ID nuevo
                "points": norm_points,  # [[x1_norm, y1_norm], ...]
                "point_labels": labels   # [1, 1, ...] (1=positivo, 0=negativo)
            })
            
            self.log(f"   Respuesta de add_prompt: {response.keys()}")
            
            mask_np = None
            # La respuesta ya contiene los outputs en 'outputs'
            if "outputs" in response:
                outputs = response["outputs"]
                if "out_obj_ids" in outputs and "out_binary_masks" in outputs:
                    out_obj_ids = outputs["out_obj_ids"]
                    out_masks = outputs["out_binary_masks"]
                    
                    # Convertir a int para comparar de forma segura
                    out_ids_int = out_obj_ids.astype(int)
                    target_id_search = int(target_obj_id) # Usamos el ID nuevo
                    
                    # Buscar el índice del obj_id
                    indices = np.where(out_ids_int == target_id_search)[0]
                    
                    if len(indices) > 0:
                        idx = indices[0]
                        mask_np = out_masks[idx]
                        self.log(f"   ✅ Máscara encontrada en índice {idx} (ID orig: {out_obj_ids[idx]})")
                    else:
                        # Si no encontramos el ID exacto, intentamos fallback pero avisando
                        self.log(f"⚠️ ID {target_id_search} no encontrado en salida {out_ids_int}. Intentando mejor coincidencia...")
                        if len(out_masks) > 0:
                            mask_np = out_masks[0] # Fallback peligroso pero funcional
                            self.log(f"   -> Usando primera máscara disponible (ID {out_ids_int[0]})")
            
            # Legacy fallback: si la estructura no es la esperada (por seguridad)
            if mask_np is None and "outputs" in response:
                 # Intentar buscar en structures antiguas o alternativas si existen
                 pass 
             
            if mask_np is None:
                # Debug info
                keys = response.keys()
                out_keys = response.get("outputs", {}).keys() if "outputs" in response else "No outputs"
                self.log(f"⚠️ Estructura recibida: {keys}, Outputs keys: {out_keys}")
                raise ValueError("No se pudo extraer máscara de la respuesta del predictor")
            
            # Asegurar dimensiones correctas
            while mask_np.ndim > 2:
                mask_np = mask_np.squeeze()

            self.log(f"   Máscara extraída con forma: {mask_np.shape}")

            # Guardar resultado (Mascara coloreada)
            filename = f"calib_{rail_type}.png"
            h, w = mask_np.shape
            cutout_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Definir colores segun el lado (Left=Green, Right=Blue)
            # Formato RGB
            if rail_type == "left":
                color = [0, 255, 0] # Verde
            else:
                color = [0, 0, 255] # Azul
            
            # Pintar pixels de la mascara
            cutout_rgba[mask_np, :3] = color
            cutout_rgba[mask_np, 3] = 180 # Alpha parcial para ver detalles debajo si se superpone
            
            save_path = os.path.join(MASKS_DIR, filename)
            Image.fromarray(cutout_rgba, mode='RGBA').save(save_path)
            
            self.log(f"✅ Riel {rail_type} generado con éxito: {filename}")
            
            return {"status": "success", "file": filename, "rail": rail_type}

        except Exception as e:
            self.log(f"❌ Error crítico en refine_rail: {traceback.format_exc()}")
            return {"status": "error", "message": f"Fallo refinando: {str(e)}"}


    # --- BUCLE PRINCIPAL ---
    def run(self):
        print(json.dumps({"type": "ready"}), flush=True)
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line: break
                
                cmd = json.loads(line)
                action = cmd.get("action")
                
                response = {"status": "error", "message": "Unknown action"}
                
                if action == "load_image":
                    response = self.load_image(cmd.get("path"))
                elif action == "segment_text":
                    response = self.segment_text(cmd.get("concepts"), cmd.get("category_types"), cmd.get("category_priors"))
                elif action == "refine_rail":
                    response = self.refine_rail(
                        cmd.get("points"), 
                        cmd.get("labels"), 
                        cmd.get("rail_type"),
                        cmd.get("obj_id")  # ✅ AHORA PASAMOS EL ID
                    )
                
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError:
                self.log("Error decodificando JSON de entrada")
            except Exception as e:
                self.log(f"Error crítico en bucle: {traceback.format_exc()}")

# --- PUNTO DE ENTRADA ---
if __name__ == "__main__":
    service = STACService()
    service.run()