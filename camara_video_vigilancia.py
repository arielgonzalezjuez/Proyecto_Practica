from typing import Union, Any

import cv2
from cv2 import Mat, CascadeClassifier
from numpy import ndarray, dtype, generic

           #Carga el clasificador pre-entrenado de deteccion de rostros.
face_cascade = cv2.CascadeClassifier('C:\\Users\\ARIEL\\PycharmProjects\\haarcascade_frontalface_default.xml')

       #Cargar el video
video_path = r'C:\Users\ARIEL\PycharmProjects\Video d Prueba\WhatsApp Video 2023-12-17 at 4.50.28 PM.mp4'
cap = cv2.VideoCapture(video_path)

        # Inicializar variables (faces,x)
faces = []
h = 0
x = 0
y = 0
w = 0

        # Verificar si la captura se abrió correctamente
if not cap.isOpened():
   print("Error al abrir el video")

while cap.isOpened():
        # Leer fotograma actal
    ret, frame = cap.read()

        # Verificar si el fotograma se ha ido leido correctamente
    if not ret:
        break

        # Convertir fotograma a escala de grises
    if ret: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en el fotograma
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30), maxSize=(200,200))

        # Imprimir los valores de las variables
    print("Frame:", frame)
    print("Gray:", gray)
    print("Faces:", faces)

        # Verificar si se detectaron rostros o no
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    else:
        print("No se detectaron rostros")

        # Dibujar un rectangulo alrededor de cada rostro detectado
for (x, y, w, h) in faces: cv2.rectangle(frame, (x, y), (x+w, y+h),
    (0, 255, 0), 2)

        # Dibujar una cuadricula
cell_size = 50
for i in range(x, x+w, cell_size):
        cv2.line(frame, (i, y),
                 (i, y+h), (255, 0, 0), 1)
        for j in range(y, y+h, cell_size):
            cv2.line(frame, (x, j), (x+w, j),
                     (255, 0, 0), 1)

        # Mostrar el fotograma resultante
if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow('Video', frame)

        # Salir del bucle si se presiona la tecla 'q'
if cv2.waitKey(1) & 0xFF == ord('q'):
       break

    # Liberar Recursos
cap.release()
cv2.destroyAllWindows()

