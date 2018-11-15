#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:15:20 2018

@author: eurekastein
"""

import psycopg2
import pandas as pd
from functools import reduce
import geopandas as gpd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from PIL import Image
import io

#Database conection server=192.168.18.22
con = psycopg2.connect(database='cadenas', user='postgres',
                       password='postgres', host='192.168.18.22')

# Define the number of steps into the productive chanin
etapas = ['etapa1', 'etapa2', 'etapa3']


#adds points manually 

def near_node(coords, tipo, engine, con):
    """Toma un punto (de algún tipo) y lo enchufa en la tabla que
        le toque, asignando el nodo de la red mas cercano.
    """
    # These should be parameters (the stages in the supply chain)
    etapas = ('etapa1', 'etapa2', 'etapa3')

    for i, coordenada in enumerate(coords):
        # cambiar las coordenaDAS de acuerdo al elemento de la lista
        print("El punto %s es %s" % (i, coordenada))
        params = {'etapas': etapa, 'coordx': coordenada[0],
                 'coordy': coordenada[1]}
        
        sql = """SELECT  id as nodos_%(etapas)s
           FROM red_vertices_pgrfrom psycopg2 import sql
           ORDER BY
            the_geom <-> ST_SetSRID(ST_MakePoint(%(coordx)d, %(coordy)d),
           32615)
           LIMIT 1
              """ % params
              
    return sql


# This funtion calculates the Dijkstra cost between points
def sql_Dcost(source_table, target_table, cost_column):
    params = {'source': source_table, 'target': target_table, 'cost': cost_column }
    qry_str = """SELECT DISTINCT ON (start_vid)
                 start_vid, end_vid, agg_cost
          FROM   (SELECT * FROM pgr_dijkstraCost(
              'select id, source, target, %(cost)s as cost from red',
              array(select distinct(closest_node) from %(source)s),
              array(select distinct(closest_node) from %(target)s),
                 directed:=false)
          ) as sub
          ORDER  BY start_vid, agg_cost asc""" % params
    return qry_str


# Append the distance values into a list
distancias = []
for i, etapa in enumerate (etapas): 
    if i < len(etapas)-1:
        print(i)
        qry_dist = sql_Dcost(etapa,etapas[i+1])
        distancia = pd.read_sql_query(qry_dist, con)
        distancias.append(distancia)

# geodataframe with geometries
prueba_1 = '/home/eurekastein/Documentos/cadenas/prueba_1.shp'

df = gpd.GeoDataFrame.from_file(prueba_1)
df["x"] =  df["geometry"].apply(lambda p: p.x)
df["y"] =  df["geometry"].apply(lambda p: p.y)
df.head()

# Merge geoDataframe with dataframes
dfs = reduce(lambda x, y: pd.merge(x, y, left_on='end_vid', right_on='start_vid', how='outer'), distancias)
dfs = df.merge(dfs, left_on='closest_no', right_on='start_vid_x')
dfs["cost"] = dfs.iloc[:, dfs.columns.str.contains('agg_cost')].sum(1)
len(dfs)

# Calculates total cost and create a matrix
matriz = dfs[["x","y","cost"]].values
matriz[:,0]

# max min values X and Y  
min_x = np.min(matriz[:,0])
max_x= np.max(matriz[:,0])
min_y= np.min(matriz[:,1])
max_y= np.max(matriz[:,1])

# GRid for interpolation
xi = np.arange(min_x, max_x, 100)
yi = np.arange(min_y, max_y, 100)
xi, yi  = np.meshgrid(xi, yi, sparse=False)
zi = griddata((matriz[:,0], matriz[:,1]),matriz[:,2],(xi,yi), method='linear')
#xi = yi = np.arange(0,1.01,0.01)

# Plot interpolation 
interpolation_rater = plt.contourf(xi,yi,zi)
interpolation_points = plt.plot(matriz[:,0],matriz[:,1],'k.')

interpolation_rater.savefig("test_rasterization.tiff", dpi=200)


#How to georeference tha raster interpolation generated

raster = '/home/eurekastein/Documentos/cadenas/raster/CEM_V3_20170619_R15_E07/Chiapas_r15m.bil'
dataset = rasterio.open(raster)













