# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Datos
data_2014=pd.read_csv("Camara 2014.csv", delimiter=";",encoding="ISO-8859-1")
data_2018=pd.read_csv("Camara 2018.csv", delimiter=";")

#Variable 1
Estad_educacion=pd.read_csv("MEN_ESTADISTICAS_EN_EDUCACION_EN_PREESCOLAR__B_SICA_Y_MEDIA_POR_DEPARTAMENTO.csv", delimiter=",",decimal=".")

#Variable 2
Estad_internet=pd.read_csv("Internet_Fijo_Penetraci_n_Departamentos.csv", delimiter=";",decimal=",")

#Variable 3
Estad_pobreza=pd.read_csv("Ind_Pob_Mult.csv", delimiter=";",decimal=".",encoding="ISO-8859-1")

#Llaves
llav_data_2014=data_2014.keys()
llav_data_2018=data_2018.keys()
llav_Estad_educacion=Estad_educacion.keys()
llav_Estad_internet=Estad_internet.keys()
llav_Estad_pobreza=Estad_pobreza.keys()

#Correccion años
ii_estad_2018=Estad_educacion[llav_Estad_educacion[0]]==2018
Estad_educacion=Estad_educacion[ii_estad_2018].sort_values(by=llav_Estad_educacion[2])

#ordenar datos a semejanza data
data_2018=data_2018.sort_values(by=llav_data_2018[0])
data_2014=data_2014.sort_values(by=llav_data_2014[0])
Estad_internet=Estad_internet.sort_values(by=llav_Estad_internet[1])
Estad_pobreza=Estad_pobreza.sort_values(by=llav_Estad_pobreza[0])

zs=Estad_pobreza[llav_Estad_pobreza[2]]
ys=Estad_educacion[llav_Estad_educacion[4]]/100
xs=Estad_internet[llav_Estad_internet[2]]
#los votos en blanco estan en miles
col=(data_2018[llav_data_2018[2]]/1000)


fig = plt.figure()
ax = fig.add_subplot(111, projection ="3d")

p= ax.scatter(xs,ys,zs,c=col, s=200)
ax.set_xlabel("Internet")
ax.set_ylabel("Educacion")
ax.set_zlabel('Pobreza')
plt.tight_layout()
ax.set_box_aspect([1,1,1])
fig.colorbar(p, shrink=1, aspect=10,ax=ax)

def shuffle(lista_a,lista_b,nombre):
    diferencia=np.mean(lista_a)-np.mean(lista_b)
    N_interracciones=10000
    lista_grande=list(lista_a)+list(lista_b)
    diferencias=np.zeros(N_interracciones)
    for i in range(N_interracciones):
        np.random.shuffle(lista_grande)
        lista_a1=lista_grande[:len(lista_a)]
        lista_b1=lista_grande[len(lista_a):]
        diferencias[i]=np.mean(lista_a1)-np.mean(lista_b1)
    p_value=2*(np.count_nonzero(diferencias>diferencia)/len(diferencias))
    plt.figure()
    plt.title("{} y P_value de {}".format(nombre,p_value))
    plt.hist(diferencias, bins=40, density="true")
    plt.vlines(diferencia,0,4,color="red")
    

    






    