import cv2
import numpy as np
import matplotlib.pyplot as plt

# Simulación de superposición de etiquetas y mejora de suavidad
def apply_label_smoothing(detections, alpha=0.3, prev_detections=None):
    """
    Aplica suavizado exponencial a las coordenadas de las detecciones para evitar saltos bruscos.
    """
    if prev_detections is None:
        return detections
    smoothed = []
    for det, prev in zip(detections, prev_detections):
        smooth = alpha * np.array(det) + (1 - alpha) * np.array(prev)
        smoothed.append(smooth.tolist())
    return smoothed

def draw_labels_no_overlap(frame, detections, labels):
    """
    Dibuja las etiquetas sin que se superpongan, desplazándolas verticalmente.
    """
    offset_y = 0
    for i, ((x1, y1, x2, y2), label) in enumerate(zip(detections, labels)):
        # Dibuja el bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
        # Calcula el texto
        text_position = (int(x1), int(y1) - 10 - offset_y)
        offset_y += 20
        # Asegura que el texto no se salga del frame
        text_position = (text_position[0], max(text_position[1], 20))
        cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# Muestra de uso
frame = np.zeros((480, 640, 3), dtype=np.uint8)
detections = [(100, 100, 200, 200), (120, 120, 220, 220)]
labels = ['Rock: 1.2kg', 'Rock: 0.8kg']
drawn = draw_labels_no_overlap(frame, detections, labels)

plt.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
plt.title("Preview with non-overlapping labels")
plt.axis('off')
plt.show()
