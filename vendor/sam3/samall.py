import os
import torch
import sam3 
import matplotlib 
matplotlib.use('Agg') # Modo sin ventanas
import matplotlib.pyplot as plt
import numpy as np # Necesario para procesar las máscaras del Generador Automático
from PIL import Image
# IMPORTANTE: Cambiamos las importaciones de "processor" y "plot_results"
from sam3.model_builder import build_sam3_image_model, SamAutomaticMaskGenerator
from sam3.visualization_utils import plot_results # La usaremos solo si se puede adaptar

# 1. Configurar el dispositivo
if torch.cuda.is_available():
    device = "cuda"
    print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("⚠️ ADVERTENCIA: Ejecutando en CPU")

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
# La imagen ya está en RGB
image_path = f"{sam3_root}/assets/images/prueba.png"

# NO NECESITAS PROMPT:
# prompt = "children, person, human, people, child, animal, train, vehicle, railroad, railway, road, street, build, house, buildings, barrier, levelcrossing, signal, signaling, semaphore"

# 2. Cargar el modelo y enviarlo a la GPU
model = build_sam3_image_model()
model.to(device)

# --- CAMBIO CLAVE: USAR EL GENERADOR AUTOMÁTICO EN LUGAR DEL PROCESSOR ---
mask_generator = SamAutomaticMaskGenerator(
    model,
    # Puedes controlar la sensibilidad aquí, si esta API lo soporta:
    # points_per_side=32, # Cuadrícula de 32x32 puntos
    # pred_iou_thresh=0.8, # Umbral de confianza para las máscaras
)

image = Image.open(image_path).convert("RGB")
print(f"📷 Imagen cargada: {image.size}")

# 3. Generación automática de máscaras (sin estado de inferencia)
masks = mask_generator.generate(image)

print(f"🔍 Se encontraron {len(masks)} objetos automáticamente.")

# El resultado 'masks' es una lista de diccionarios, no el objeto 'output' del text processor.
# Si tu plot_results no acepta esta nueva estructura, debemos escribir una visualización manual.
# Por ahora, veamos si la función original acepta la lista de diccionarios (es poco probable):
try:
    # Esto es una suposición. Si falla, el script sigue abajo.
    plot_results(image, masks) 
except Exception as e:
    print(f"⚠️ La función plot_results falló con el generador automático: {e}")
    print("-> Visualizando manualmente.")
    
    # 4. Visualización manual (Lo más seguro)
    # Convertimos la lista de diccionarios de máscaras a un formato que podamos pintar:
    if len(masks) > 0:
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        # Tomamos solo las áreas grandes, las más pequeñas suelen ser ruido
        mask_array = np.stack([m['segmentation'] for m in sorted_masks])
        
        # Crear una imagen vacía y pintar las máscaras
        segmentation_image = np.zeros_like(image)
        for i, mask in enumerate(mask_array):
            color = np.random.randint(0, 256, 3, dtype=np.uint8)
            segmentation_image[mask] = color
            
        plt.imshow(image)
        plt.imshow(segmentation_image, alpha=0.6)
        plt.axis('off')

# Guardar el resultado (ya sea de plot_results o manual)
output_filename = "resultado_automatico.png"
plt.savefig(output_filename)
print(f"✅ Imagen guardada como: {output_filename}")