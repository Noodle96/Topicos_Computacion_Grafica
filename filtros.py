import numpy as np
import cv2

#   Función que aplica un filtro n x n y calcula la mediana en cada pixel de una imagen.
def mediana_convolucion_nxn(imagen, filtro, n):
  if isinstance(imagen, str):
    imagen = cv2.imread(imagen)

  # Validar la imagen
  if imagen is None:
    raise ValueError('Error al leer la imagen')

  # Verificar si es imagen a color
  if len(imagen.shape) > 2:
    raise ValueError('La imagen debe ser en escala de grises')

  # Calcular el padding necesario
  pad = (n - 1) // 2

  pad_width = np.array([(pad, pad), (pad, pad)]) 

  # Añadir padding a la imagen
  imagen_pad = np.pad(imagen, pad_width, mode='constant')

  # Inicializar la imagen filtrada con medianas
  imagen_mediana = np.zeros_like(imagen)

  # Recorrer cada pixel de la imagen
  for i in range(imagen.shape[0]):
    for j in range(imagen.shape[1]):
      # Extraer el sub-bloque de la imagen con padding
      sub_bloque = imagen_pad[i:i+n, j:j+n]

      # Aplicar la convolución con el filtro
      convolucion = np.sum(sub_bloque * filtro, axis=(0, 1))

      # Calcular la mediana del sub-bloque
      mediana = np.median(sub_bloque, axis=(0, 1))  # Asume un valor único

      # Asignar la mediana a la imagen filtrada
      imagen_mediana[i, j] = mediana  # Extraer el valor único para grayscale

  return imagen_mediana

# FILTRO DE SOBEL
# Función que implementa una versión simplificada del filtro de Sobel.
def sobel(imagen):
    # Convertir la imagen a un array de NumPy
    imagen_np = np.array(imagen)

    # Inicializar matrices para almacenar las derivadas (tamaño reducido)
    derivada_x = np.zeros((imagen_np.shape[0] - 2, imagen_np.shape[1] - 2))
    derivada_y = np.zeros((imagen_np.shape[0] - 2, imagen_np.shape[1] - 2))

    # Matrices de filtro Sobel
    filtro_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filtro_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Recorrer los píxeles de la imagen (excluyendo los bordes)
    for i in range(1, imagen_np.shape[0] - 1):
        for j in range(1, imagen_np.shape[1] - 1):
            # Extraer el parche de la imagen actual
            parche = imagen_np[i-1:i+2, j-1:j+2]

            # Aplicar la convolución con los filtros Sobel
            derivada_x[i - 1, j - 1] = np.sum(parche * filtro_x)
            derivada_y[i - 1, j - 1] = np.sum(parche * filtro_y)

    # Calcular la magnitud del gradiente
    magnitud = np.sqrt(derivada_x**2 + derivada_y**2)

    # Normalizar la magnitud a un rango de 0 a 255
    magnitud_normalizada = (magnitud - magnitud.min()) / (magnitud.max() - magnitud.min()) * 255

    # Convertir la imagen resultante a un tipo de dato adecuado para mostrar
    imagen_resultante = magnitud_normalizada.astype('uint8')

    return imagen_resultante

# FILTRO DE ROBERTS
# Función que implementa una versión simplificada del filtro Roberts Cross Operator.
def roberts(imagen):
  # Convertir la imagen a un array de NumPy
  imagen_np = np.array(imagen)
  # Inicializar matrices para almacenar las derivadas (tamaño reducido)
  derivada_x = np.zeros((imagen_np.shape[0] - 2, imagen_np.shape[1] - 2))
  derivada_y = np.zeros((imagen_np.shape[0] - 2, imagen_np.shape[1] - 2))
  # Matrices de filtro Roberts
  filtro_x = np.array([[-1, 0], [0, 1]])
  filtro_y = np.array([[0, -1], [1, 0]])

  # Recorrer los píxeles de la imagen (excluyendo bordes)
  for i in range(1, imagen_np.shape[0] - 1):
    for j in range(1, imagen_np.shape[1] - 1):
      # Extraer el parche de la imagen actual (sin padding)
      parche_x = imagen_np[i:i + 2, j:j + 2]
      parche_y = imagen_np[i:i + 2, j:j + 2]

      # Aplicar la convolución con los filtros
      derivada_x[i - 1, j - 1] = np.sum(parche_x * filtro_x)
      derivada_y[i - 1, j - 1] = np.sum(parche_y * filtro_y)

  # Calcular la magnitud del gradiente
  magnitud = np.sqrt(derivada_x**2 + derivada_y**2)

  # Normalizar la magnitud a un rango de 0 a 255
  magnitud_normalizada = (magnitud - magnitud.min()) / (magnitud.max() - magnitud.min()) * 255

  # Convertir la imagen resultante a un tipo de dato adecuado para mostrar
  imagen_resultante = magnitud_normalizada.astype('uint8')

  return imagen_resultante

# Función que implementa una versión simplificada del filtro Prewitt para detección de bordes.
def prewitt(imagen):
    # Convertir la imagen a un array de NumPy
    imagen_np = np.array(imagen)

    # Inicializar matrices para almacenar las derivadas (tamaño reducido)
    derivada_x = np.zeros((imagen_np.shape[0] - 2, imagen_np.shape[1] - 2))
    derivada_y = np.zeros((imagen_np.shape[0] - 2, imagen_np.shape[1] - 2))

    # Matrices de filtro Prewitt
    filtro_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filtro_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Recorrer los píxeles de la imagen (excluyendo los bordes)
    for i in range(1, imagen_np.shape[0] - 1):
        for j in range(1, imagen_np.shape[1] - 1):
            # Extraer el parche de la imagen actual
            parche = imagen_np[i-1:i+2, j-1:j+2]

            # Aplicar la convolución con los filtros Prewitt
            derivada_x[i - 1, j - 1] = np.sum(parche * filtro_x)
            derivada_y[i - 1, j - 1] = np.sum(parche * filtro_y)

    # Calcular la magnitud del gradiente
    magnitud = np.sqrt(derivada_x**2 + derivada_y**2)

    # Normalizar la magnitud a un rango de 0 a 255
    magnitud_normalizada = (magnitud - magnitud.min()) / (magnitud.max() - magnitud.min()) * 255

    # Convertir la imagen resultante a un tipo de dato adecuado para mostrar
    imagen_resultante = magnitud_normalizada.astype('uint8')

    return imagen_resultante



# MAIN
imagen_a_color = cv2.imread('img/mandrill.jpg')  # Leer imagen a color
imagen = cv2.cvtColor(imagen_a_color, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises

filtro_3x3 = np.array([
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1]
])

filtro_5x5 = np.array([
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1]
]) 

filtro_7x7 = np.array([
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
]) 

filtro_9x9 = np.array([
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
]) 

filtro_11x11 = np.array([
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]) 

imagen_mediana_3x3 = mediana_convolucion_nxn(imagen, filtro_3x3, 3)
imagen_mediana_5x5 = mediana_convolucion_nxn(imagen, filtro_5x5, 5)
imagen_mediana_7x7 = mediana_convolucion_nxn(imagen, filtro_7x7, 7)
imagen_mediana_9x9 = mediana_convolucion_nxn(imagen, filtro_9x9, 9)
imagen_mediana_11x11 = mediana_convolucion_nxn(imagen, filtro_11x11, 11)
imagen_filtro_sobel = sobel(imagen)
imagen_filtro_roberts = roberts(imagen)
imagen_filtro_prewitt = prewitt(imagen)


#OUTPUT DE LAS IMAGENES
# Mostrar la imagen original y las imágenes filtradas
import matplotlib.pyplot as plt

plt.imshow(imagen, cmap='gray')
plt.title('Imagen original en escala de grises')
plt.show()

plt.imshow(imagen_mediana_3x3, cmap='gray')
plt.title('Imagen filtrada con mediana (3x3)')
plt.show()

plt.imshow(imagen_mediana_5x5, cmap='gray')
plt.title('Imagen filtrada con mediana (5x5)')
plt.show()

plt.imshow(imagen_mediana_7x7, cmap='gray')
plt.title('Imagen filtrada con mediana (7x7)')
plt.show()

plt.imshow(imagen_mediana_9x9, cmap='gray')
plt.title('Imagen filtrada con mediana (9x9)')
plt.show()

plt.imshow(imagen_mediana_11x11, cmap='gray')
plt.title('Imagen filtrada con mediana (11x11)')
plt.show()

plt.imshow(imagen_filtro_sobel, cmap='gray')
plt.title('Imagen filtrada con sobel')
plt.show()

plt.imshow(imagen_filtro_roberts, cmap='gray')
plt.title('Imagen con filtro Roberts')
plt.show()

plt.imshow(imagen_filtro_prewitt, cmap='gray')
plt.title('Imagen con filtro prewitt')
plt.show()
