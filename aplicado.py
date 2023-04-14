import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Cargar el modelo entrenado
model = load_model('modelo_entrenado.h5')

# Cargar las etiquetas de las clases
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preparar la imagen para la inferencia
    img = cv2.resize(frame, (32, 32))
    img = img.astype('float32') / 255
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Realizar la inferencia y obtener la clase más probable
    predictions = model.predict(img)
    class_idx = np.argmax(predictions[0])
    label = class_labels[class_idx]

    # Mostrar la etiqueta de clase en la ventana
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Reconocimiento de objetos', frame)

    # Salir del bucle cuando se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
