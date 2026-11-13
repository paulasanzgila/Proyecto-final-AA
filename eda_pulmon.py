import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Cargar el dataset en tu archivo local
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

sns.set_theme(style="whitegrid")

# 1. Distribución de tumores con un Bar Chart
fig0, ax0 = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x='LUNG_CANCER', palette=['#66b3ff', '#ff9999'], ax=ax0)
ax0.set_title('Casos de Cáncer (NO vs YES)')
plt.tight_layout()
plt.show()

# 2. Histograma de Edad (Variable numérica principal)
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(data=df, x='AGE', hue='LUNG_CANCER', kde=True, ax=ax1, palette=['#66b3ff', '#ff9999'])
ax1.set_title('Distribución de la Edad según el Diagnóstico')
plt.tight_layout()
plt.show()

# 3. Boxplots de TODOS los síntomas
# Seleccionamos todas las variables que son síntomas (excluyendo Edad, Género y el target)
columnas_sintomas = [col for col in df.columns if col not in ['GENDER', 'AGE', 'LUNG_CANCER']]

# Creamos una cuadrícula de 4x4 para los 13 síntomas (sobran 3 huecos)
fig2, axes2 = plt.subplots(4, 4, figsize=(16, 14))
fig2.subplots_adjust(hspace=0.6, wspace=0.4) # Espacio para que no se solapen los títulos
fig2.suptitle("Comparativa de TODOS los Síntomas según el Diagnóstico", fontsize=18, y=1.02)

axes2 = axes2.flatten()

for i, col in enumerate(columnas_sintomas):
    sns.boxplot(data=df, x='LUNG_CANCER', y=col, ax=axes2[i], palette=['#66b3ff', '#ff9999'])
    # Formatear el título (quitar guiones bajos y poner mayúscula)
    axes2[i].set_title(col.replace('_', ' ').title(), fontsize=11)
    axes2[i].set_xlabel('') 
    axes2[i].set_ylabel('')

# Ocultar los huecos sobrantes de la cuadrícula (del 14 al 16)
for j in range(len(columnas_sintomas), 16):
    fig2.delaxes(axes2[j])

plt.tight_layout()
plt.show()

# 4. Heatmap de Correlación (Opcional pero muy recomendado)
# Para ver qué variables tienen más relación con el cáncer
fig3, ax3 = plt.subplots(figsize=(12, 10))
df_corr = df.copy()
# Mapeamos temporalmente para que el heatmap funcione
df_corr['GENDER'] = df_corr['GENDER'].map({'M': 1, 'F': 0})
df_corr['LUNG_CANCER'] = df_corr['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
for col in columnas_sintomas:
    if df_corr[col].max() == 2:
        df_corr[col] = df_corr[col].map({2: 1, 1: 0})

corr_matrix = df_corr.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax3, vmin=-1, vmax=1)
ax3.set_title('Matriz de Correlación de Variables Clínicas')
plt.tight_layout()
plt.show()

# 3. Variable de Salida a Binario y transformación

df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

for col in df.columns:
    if col not in ['GENDER', 'AGE', 'LUNG_CANCER']:
        if df[col].max() == 2:
            df[col] = df[col].map({2: 1, 1: 0})

# 4. Frecuencias relativas y balanceo
print("\nPROPORCIONES ORIGINALES")
frecuencias = df['LUNG_CANCER'].value_counts(normalize=True) * 100
print(f"Proporción de 0 (No Cáncer): {frecuencias[0]:.2f}%")
print(f"Proporción de 1 (Cáncer): {frecuencias[1]:.2f}%")

# Calcular pesos para balancear
pesos = compute_class_weight(class_weight='balanced', classes=np.unique(df['LUNG_CANCER']), y=df['LUNG_CANCER'])
diccionario_pesos = {0: pesos[0], 1: pesos[1]}
print(f"\nPesos calculados para balancear: {diccionario_pesos}")

# Para crear el nuevo dataset con los cambios realizados
#df.to_csv("dataset_limpio_pulmon.csv", index=False)
