# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Datos
data_2014=pd.read_csv("Camara 2014.csv", delimiter=";",encoding="ISO-8859-1")
data_2018=pd.read_csv("Camara 2018.csv", delimiter=";")

Estad_educacion=pd.read_csv("MEN_ESTADISTICAS_EN_EDUCACION_EN_PREESCOLAR__B_SICA_Y_MEDIA_POR_DEPARTAMENTO.csv", delimiter=",",decimal=".")

#Llaves
llav_data_2014=data_2014.keys()
llav_data_2018=data_2018.keys()
llav_Estad_educacion=Estad_educacion.keys()

#Correccion años
ii_estad_2018=Estad_educacion[llav_Estad_educacion[0]]==2018
Estad_educacion=Estad_educacion[ii_estad_2018]





#filtrado datos

def organizacion(data1,data2):
    len(data1[::1])
    