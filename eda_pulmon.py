import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Cargar el dataset en mi archivo local
df = pd.read_csv(r"C:\3 de carrera\aprendizaje automatico\proyecto final\survey lung cancer.csv")

# 1. Limpieza de datos
print(df.columns)

# El dataset tiene un espacio al final del nombre de algunas columnas, lo limpiamos
df.columns = df.columns.str.strip()

# No hay columnas de ID ni columnas vacías, pero vamos a comprobar nulos
nulos = df.isnull().sum()
print("Columnas con valores nulos:\n\n", nulos[nulos>0])
# no hay ninguna columna con valores faltantes


# 2. Creación de Diagramas

# 1. Distribución de tumores con un Bar Chart
fig0, ax0 = plt.subplots(figsize=(5, 5))
sns.countplot(data=df, x='LUNG_CANCER', palette=['#66b3ff', '#ff9999'], ax=ax0)
ax0.set_title('Casos de Cáncer (NO vs YES)')

# 2. Figura de Histogramas y Scatter Plots
fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
fig1.suptitle("Distribuciones y Relaciones según el Diagnóstico")

# Histograma de Edad
sns.histplot(data=df, x='AGE', hue='LUNG_CANCER', kde=True, ax=axes1[0, 0], palette=['#66b3ff', '#ff9999'])
axes1[0, 0].set_title('Distribución de la Edad')

# Gráfico de Fumadores
sns.countplot(data=df, x='SMOKING', hue='LUNG_CANCER', ax=axes1[0, 1], palette=['#66b3ff', '#ff9999'])
axes1[0, 1].set_title('Fumadores vs Diagnóstico')

# Scatter Plot Edad vs Ansiedad
sns.scatterplot(data=df, x='AGE', y='ANXIETY', hue='LUNG_CANCER', ax=axes1[1, 0], palette=['#66b3ff', '#ff9999'])
axes1[1, 0].set_title('Relación Edad vs Ansiedad')

# Scatter Plot Edad vs Fatiga
sns.scatterplot(data=df, x='AGE', y='FATIGUE', hue='LUNG_CANCER', ax=axes1[1, 1], palette=['#66b3ff', '#ff9999'])
axes1[1, 1].set_title('Relación Edad vs Fatiga')


# 3. Figura de Boxplots
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
fig2.suptitle("Síntomas según el Diagnóstico")

# Edad
sns.boxplot(data=df, x='LUNG_CANCER', y='AGE', ax=axes2[0, 0], palette=['#66b3ff', '#ff9999'])
axes2[0, 0].set_title("Edad según Diagnóstico")

# Tos
sns.boxplot(data=df, x='LUNG_CANCER', y='COUGHING', ax=axes2[0, 1], palette=['#66b3ff', '#ff9999'])
axes2[0, 1].set_title("Tos según Diagnóstico")

# Dificultad para respirar
sns.boxplot(data=df, x='LUNG_CANCER', y='SHORTNESS OF BREATH', ax=axes2[1, 0], palette=['#66b3ff', '#ff9999'])
axes2[1, 0].set_title("Dificultad para Respirar")

# Dolor en el pecho
sns.boxplot(data=df, x='LUNG_CANCER', y='CHEST PAIN', ax=axes2[1, 1], palette=['#66b3ff', '#ff9999'])
axes2[1, 1].set_title("Dolor en el Pecho")

plt.show()

# Como hemos observado, hay variables con 1 y 2 (YES/NO). Para evitar una fuga de datos, la estandarización se hará en el archivo MLP donde se encontrará todo el código que entrenará la red neuronal.

# 3. Variable de Salida a Binario y transformación de datos
df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Transformamos las columnas que tienen 1 y 2 a 0 y 1
for col in df.columns:
    if col not in ['GENDER', 'AGE', 'LUNG_CANCER']:
        if df[col].max() == 2:
            df[col] = df[col].map({2: 1, 1: 0})

# 4. Frecuencias relativas y balanceo
print("\nPROPORCIONES ORIGINALES")
frecuencias = df['LUNG_CANCER'].value_counts(normalize=True) * 100
print(f"Proporción de 0 (No Cáncer): {frecuencias[0]:.2f}%")
print(f"Proporción de 1 (Cáncer): {frecuencias[1]:.2f}%")

# Calcular pesos para balancear (para que el 1 no sea peor que el 0)
pesos = compute_class_weight(class_weight='balanced', classes=np.unique(df['LUNG_CANCER']), y=df['LUNG_CANCER'])
diccionario_pesos = {0: pesos[0], 1: pesos[1]}
print(f"\nPesos calculados para balancear: {diccionario_pesos}")

# Para crear el nuevo dataset con los cambios realizados
#df.to_csv("dataset_limpio_pulmon.csv", index=False)