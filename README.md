# Clasificación de Resonancias Magnéticas Cerebrales

Un proyecto de aprendizaje profundo para clasificar resonancias magnéticas cerebrales y detectar tumores usando TensorFlow.

## Conjunto de Datos

El conjunto de datos utilizado en este proyecto es [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data) de Kaggle. Contiene:

- 253 resonancias magnéticas cerebrales en total
- 155 resonancias con tumores (casos positivos)
- 98 resonancias sin tumores (casos negativos)
- Todas las imágenes están en formato JPG

El conjunto de datos está organizado en dos carpetas:
- `yes/`: Contiene resonancias magnéticas con tumores
- `no/`: Contiene resonancias magnéticas sin tumores

Cada resonancia es una imagen en escala de grises que muestra una vista transversal del cerebro. Las imágenes han sido preprocesadas y se ha eliminado el cráneo para enfocarse en el tejido cerebral donde pueden estar presentes los tumores.

## Estructura del Proyecto

```
.
├── artifacts/          # Artefactos generados
│   ├── models/        # Modelos guardados
│   └── plots/         # Gráficos y visualizaciones generadas
├── data/              # Directorio de datos
│   ├── processed/     # Conjunto de datos procesado
│   └── raw/          # Conjunto de datos sin procesar
│       ├── yes/      # Resonancias con tumores
│       └── no/       # Resonancias sin tumores
├── notebooks/         # Notebooks de Jupyter
├── src/              # Código fuente
│   ├── data/         # Carga y preprocesamiento de datos
│   ├── models/       # Arquitectura del modelo
│   ├── utils/        # Funciones de utilidad
│   └── main.py       # Script principal
└── requirements.txt   # Dependencias del proyecto
```

## Configuración

1. Crear un entorno virtual:
```bash
conda create -n mri-classification python=3.8
conda activate mri-classification
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Ejecutar el pipeline de entrenamiento:
```bash
python src/main.py
```

## Arquitectura del Modelo

El modelo utiliza una arquitectura CNN con:
- 4 capas convolucionales
- Capas de max pooling
- Dropout para regularización
- Capas densas para clasificación

## Características de Rendimiento

- Soporte para entrenamiento multi-GPU con MirroredStrategy
- Entrenamiento con precisión mixta para cálculos más rápidos
- Carga y preprocesamiento de datos en paralelo
- ThreadPoolExecutor para predicciones en paralelo
- Optimización automática de hardware

## Resultados

El modelo alcanza:
- 75% de precisión para casos sin tumor
- 71% de precisión para casos con tumor
- 73% de precisión general

Los resultados se guardan en:
- Pesos del modelo: `artifacts/models/best_model.h5`
- Gráficos de entrenamiento: `artifacts/plots/training_history.png`
![Gráficos de entrenamiento](artifacts/plots/training_history.png)
- Matriz de confusión: `artifacts/plots/confusion_matrix.png`
![Matriz de confusión](artifacts/plots/confusion_matrix.png)


## Créditos

Este proyecto está basado en el notebook de Jupyter de [Anirudh Bansal](https://www.kaggle.com/anibansal) de su kernel de Kaggle [Brain MRI Classification](https://www.kaggle.com/code/anibansal/brain-mri-classification). El trabajo original ha sido reestructurado en un paquete Python apropiado con mejor organización y modularidad; y se optimizaron varios procesos para mejorar el rendimiento y la velocidad, usando técnicas de paralelización y optimización de hardware.