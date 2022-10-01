#Librerias
from importlib.resources import path
from json import load
import tensorflow as tf #Libreria de tensorflow usada para la IA
import numpy as np #Libreria de numpy usada para arrays y calculos
import matplotlib.pyplot as plt #Lubreria de matplotlib para hacer diagramas
import os
import pathlib
from keras.models import load_model

#Verificaremos si ya tenemos una red entrenada

if os.path.exists("./modelos/") == False:
    os.makedirs("./modelos/")
    #Datos de entrenamiento
    """
    El primer paso es darle los datos con los que queremos trabajar, en mi caso le he dado
    7 valores en horas y sus respectivos equivalentes en minutos para que empiece a trabajar en base a esto
    y descubra el metodo para convertir de horas a minutos

    """
    horas = np.array([1, 5, 10, 15, 20, 50, 100 ], dtype=float)
    minutos = np.array([60,300,600,900,1200,3000,6000], dtype=float)

    #Ahora generamos la neurona que hara el entrenamiento
    capa = tf.keras.layers.Dense(units=1, input_shape=[1])
    modelo = tf.keras.Sequential([capa])

    #Compilaremos el modelo y le diremos cuanta sera nuestra tolerancia a errores
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.09),
        loss="mean_squared_error"
    )

    """
    Ahora le diremos cuantos entrenamientos debe realizar y mostraremos una grafica con los cambios
    que han habido en cada etapa, esto ayudara a refinar
    """
    print("Entrenando...")
    historial = modelo.fit(horas, minutos, epochs=7050, verbose=False) #Aqui diremos los valores y la cantidad de veces
    print("Entrenamiento completado")
    plt.xlabel("# Epoca")
    plt.ylabel("Magnitud de perdida")
    plt.plot(historial.history["loss"])

    #Guardemos el modelo ya entrenado
    modelo.save("./modelos/red.h5")

#En caso de que si exista la cargaremos
else:
    modelo = load_model("./modelos/red.h5")
    print("Modelo cargado correctamente")



#Aqui va el cuerpo del programa estandar
rotoBucle = False
while rotoBucle != True:
    print("Por favor introduzca una cantidad de horas: (Si desea salir escriba salir")
    horasEntrada = input("Horas: ")
    if horasEntrada.lower() == "salir":
        rotoBucle = True
    elif horasEntrada.isnumeric() == True:
        horasEntrada = float(horasEntrada)
        resultado = np.round(modelo.predict([horasEntrada]))
        print(print("El resultado es: " + str(resultado) + " minutos!"))
        if resultado > horasEntrada * 60:
            print("Esta prediccion se ha pasado")
        elif resultado < horasEntrada * 60:
            print("Esta prediccion se ha quedado corta")
        else:
            print("Esta prediccion ha sido exacta")
    else:
        print("Por favor introduzca un valor numerico")

