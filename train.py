import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# cargar dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# limpiar columnas (MUY IMPORTANTE)
df.columns = df.columns.str.strip()

# verificar
print("Columnas:", df.columns)

# función objetivo
def categoria(nota):
    if nota >= 15:
        return "Alto"
    elif nota >= 10:
        return "Medio"
    else:
        return "Bajo"

# asegurar que G3 existe
print("Existe G3:", "G3" in df.columns)

df["target"] = df["G3"].apply(categoria)

# variables
X = df[["studytime", "failures", "absences"]]
y = df["target"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# modelo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# guardar
joblib.dump(model, "models/model.pkl")

print("Modelo entrenado correctamente")