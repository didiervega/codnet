# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:50:41 2019

@author: Didier Vega-Oliveros
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os,glob
from sklearn.neighbors import NearestNeighbors

from mpl_toolkits.basemap import Basemap
from pathlib import Path
import parallel

EARTH_RADIUS = 6371 # km
degree_to_rad = float(np.pi / 180.0)

owd = os.getcwd()
datafolder = os.path.join(os.getcwd(),'data')
datafile = os.path.join(datafolder,'conf_dat.csv')


'''''
    This method returns the path to the graphs folder.
    If it does not exist, it will be created
'''''
def get_graphsPath(graphsPath=os.path.join(owd, 'graphs')):
    Path(graphsPath).mkdir(parents=True, exist_ok=True)
    return graphsPath
        
        
def func_sum(x):    
    lx = list(x)    
    sumX = 0;
    for i in lx:               
        sumX += i['weight']
    return {'weight': sumX}


def getFilePattern(neighbors=0,timeWindow='year',MAX_DISTANCE=500, ext='gml', begin='net-year',graphsPath=os.path.join(owd, 'graphs')):
    fPattern = ''
    if neighbors == 0:
        fPattern = os.path.join(get_graphsPath(graphsPath), begin+"*MAX_DISTANCE_"+str(MAX_DISTANCE)+'_'+ timeWindow+'_.'+ext)
    else:
        fPattern = os.path.join(get_graphsPath(graphsPath), begin+"*neighbors_"+str(neighbors)+'_MAX_DISTANCE_'+str(MAX_DISTANCE)+'_'+ timeWindow+'_.'+ext)
        
    return fPattern

    
def getFileName(begin='net-year',neighbors=0,timeWindow='year',MAX_DISTANCE=500,
                ext='gml',path=''):
    
    if path=='':
        path = get_graphsPath()
    else:
        path = get_graphsPath(graphsPath=path)
        
    if neighbors > 0:
        return os.path.join(path,begin+"_neighbors_"+str(neighbors)+'_MAX_DISTANCE_'+str(MAX_DISTANCE)+'_'+ timeWindow+'_.'+ext)
    else:
        return os.path.join(path,begin+'_MAX_DISTANCE_'+str(MAX_DISTANCE)+'_'+ timeWindow+'_.'+ext)
    
         
def getMonthData (conf, month, neighbors,event, MAX_DISTANCE):        
    
    if month == -1:
        df0 = conf.copy()
    else:    
        df0 = conf.loc[(conf['month'] == month)].copy()
        
    df0.drop_duplicates(subset=['gid'], inplace=True)
    dfe = df0[['rlatitude','rlongitude']]
    nbrs = distances = indices = 0
    nbrs = NearestNeighbors(n_neighbors=neighbors+1, algorithm = 'ball_tree',metric='haversine')
    nbrs.fit(dfe)

    ## for getting the graph nbrs.kneighbors_graph(dfe).toarray()
    distances, indices = nbrs.kneighbors(dfe)
    distances = distances*EARTH_RADIUS

    #constructing the graph
    G=nx.MultiGraph()
    edges = list()
    
    for ix in range(len(indices)):
        row = df0.iloc[ix]
        name = int(row.gid)    
        G.add_node(name)
        G.nodes[name]['lat'] = row.latitude
        G.nodes[name]['lng'] = row.longitude
        G.nodes[name]['event'] = int(row[event])
        G.nodes[name]['month'] = int(row.month)
        G.nodes[name]['year'] = int(row.year)
                
        ## not considering self-loops
        for ex in range(1,neighbors+1):
            neig = int(df0.iloc[indices[ix][ex]].gid)        
            if name != neig and distances[ix][ex] <= MAX_DISTANCE:
                ed = (name, neig,{'weight': 1, 'month':int(row.month)})
                edges.append(ed)
            
             
    G.add_edges_from(edges)    
    edges = list()
    
    isolated = list(nx.isolates(G))    
    if len(isolated) > 0:
        print(len(isolated),'isolated nodes.')
        for node in isolated:        
            ed = (node, node,{'weight': 1,'month':0})             
            edges.append(ed)
        G.add_edges_from(edges)            
        
    E = list(G.edges(data=True)) 
    N = list(G.nodes(data=True))
    del G, df0, dfe, distances,indices
    
    return E,N 


 
## Prints the conventional white figure map using different colors for the components
def drawGraph(Gy, figname):
    lat = list(Gy.nodes(data = 'lat'))
    long = list(Gy.nodes(data = 'lng'))
    df = pd.DataFrame() 
    df['id'] = pd.DataFrame(lat)[0] # ID label in pos 0               
    df['lat'] = pd.DataFrame(lat)[1]
    df['lng'] = pd.DataFrame(long)[1]                
    df['degree'] = pd.DataFrame(Gy.degree)[1]
    comps = sorted(nx.connected_components(Gy), key = len, reverse=True)
    cid = 0
    for comp in comps:
        compl = list(comp)
        df.loc[df['id'].isin(compl),'comp'] = cid
        cid +=1
    
    pos = dict()
    for row in df.itertuples():
        pos[row.id] = [row.lng, row.lat]
      
    fig = plt.figure(figsize=(15, 15), edgecolor='w')     
    m = Basemap(resolution='c', 
            llcrnrlat=-60, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, )
    m.drawcoastlines()
    m.drawcountries()   
    
    # Remove self-loops
    self_loops = list(nx.selfloop_edges(Gy))
    Gy.remove_edges_from(self_loops)
    
    nx.draw_networkx_edges(Gy,pos=pos,edge_color='blue', alpha=.5, width=1.0)
    m.scatter(df['lng'].values, df['lat'].values, latlon=True,
                s=25.0, c= df['comp'].values,
              cmap='tab20', alpha=0.4, marker='o', edgecolor='k')
    plt.savefig(figname, format='png', bbox_inches='tight')
    plt.close()
    
    
def constFunc(years, args):
    conf, timeWindow,neighbors,event, MAX_DISTANCE = args
    for year in years:
        df0 =  conf.loc[(conf['year'] == year)]
        Gy = nx.Graph()
        dfy = pd.DataFrame()
        
        if timeWindow == 'month':
            
            for month in range(1,13):       
                Ey, Ny = getMonthData(df0, month,neighbors,event,MAX_DISTANCE)        
                Gy.add_nodes_from(Ny)       
                dfm = pd.DataFrame(Ey)
                dfy = pd.concat([dfy, dfm], ignore_index=True, sort=False)
        else:
            Ey, Ny = getMonthData(df0, -1, neighbors,event,MAX_DISTANCE)        
            Gy.add_nodes_from(Ny)       
            dfy = pd.DataFrame(Ey)
            
        dfy = dfy.groupby([dfy[0],dfy[1]]).agg({0: 'first',
                                                    1: 'first',
                                                    2: func_sum})
    
        edges = [list(edge) for edge in list(dfy.values)]
        Gy.add_edges_from(edges)  
        
        filename = getFileName(begin='net-year_'+str(year),neighbors=neighbors,
                               timeWindow=timeWindow,
                               MAX_DISTANCE=MAX_DISTANCE, ext='gml')
        
        figname = getFileName(begin='fignet-year_'+str(year),neighbors=neighbors,
                              timeWindow=timeWindow,
                              MAX_DISTANCE=MAX_DISTANCE, ext='png')
        nx.write_gml(Gy, filename)
        drawGraph(Gy, figname)
    
    return []

def getConfEventData(file=datafile,event='state'):
    conf = pd.read_csv(file)
    conf.drop(columns=['Unnamed: 0', 'ccode'], inplace = True)
    conf = conf.loc[(conf[event] > 0)]
    conf.loc[:,'rlatitude'] = np.deg2rad(conf['latitude'])
    conf.loc[:,'rlongitude'] = np.deg2rad(conf['longitude'])
    return conf
    

def contructYearsNetworks(event='state',neighbors=3, MAX_DISTANCE=500, timeWindow = 'month'):
      
    conf = getConfEventData(event=event)
    years = np.sort(conf['year'].unique())
    lyears = years.tolist()    
    lFull = parallel.For(lyears,constFunc,argsF=(conf, timeWindow,neighbors,event, MAX_DISTANCE)) 
   
 
if __name__ == "__main__":

    event = 'state'    
    neighbors = [3,7,11]
    MAX_DISTANCE = [100,500,EARTH_RADIUS] # km
    timeWindow = ['month','year']     
    
    for k in neighbors:
        for distance in MAX_DISTANCE:
            for twindow in timeWindow:
                contructYearsNetworks(event,k,distance,timeWindow = twindow)    
                
                
    
 