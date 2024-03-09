# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:50:41 2019

@author: Didier Vega-Oliveros
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os, sys, glob,getopt
from itertools import chain
import graphConstruction as gc
from scipy.stats import moment
import igraph as ig
import leidenalg as la
import parallel
from mpl_toolkits.basemap import Basemap
sys.stdout.flush()

owd = os.getcwd()

class codnet:
    
    def __init__(self):
    
        self.EARTH_RADIUS = gc.EARTH_RADIUS # km
        self.folderData = gc.get_graphsPath()

        
    def complexity(self,Z):
        return moment(Z, moment=2)/(np.mean(Z)**2)
        
    def getValidConfNodes(self,dfPrec,dfTemp, conf):
        sP = set(dfPrec.index)
        sT = set(dfTemp.index)
        sC = set(conf.index)
        
        validIds = sP & sT & sC
        return conf.loc[validIds]  

    def _convertNxtoIg(self, G):
        nx.write_gml(G, 'temp.gml')
        Gig = ig.read('temp.gml',format="gml") # Create new IG graph from file
        return Gig
    
    #measure=la.ModularityVertexPartition
    def _setGraphCommunityLabels(self, G, measure=la.RBConfigurationVertexPartition,
                          min_comm_size=5):
        
        Gig = self._convertNxtoIg(G)
        partition = la.find_partition(Gig, measure, n_iterations=-1, resolution_parameter=2)        
        ## optimizer.consider_empty_community=False
        optimizer = la.Optimiser()        
        diff = 1
        while diff > 0:
            diff = optimizer.optimise_partition(partition, n_iterations=-1)
        
        dicMember = {x['label']:int(partition.membership[i]) for i,x in enumerate(Gig.vs)}
        nx.set_node_attributes(G, dicMember, 'cLabel')       
        
        return G
   
    #measure=la.ModularityVertexPartition
    def getCommunityLabel(self,fileGML='net-ALL-years_neighbors_3_MAX_DISTANCE_500_year_.gml', 
                          min_comm_size=5):
                
        Gy = nx.read_gml(fileGML)       
        comm = pd.DataFrame(dict(Gy.nodes(data=True))).T
        partition = comm['cLabel'].value_counts()
        
        #removing the communities with less than 5 events over the years
        partition =  partition[partition >= min_comm_size]
        
        
        comm = comm.loc[comm['cLabel'].isin(list(partition.index))]   
        comm.drop(columns=['event','month','year'], inplace = True)  
        comm['Label'] = comm.index.astype(int)
        comm['cLabel'] = comm['cLabel'].astype(int).astype(str)
       
        return comm
           
    
    def contructConsolidateYearNetwork(self,neighbors=3,MAX_DISTANCE=500,timeWindow = 'year'):
        
        data_folder = gc.getFilePattern(neighbors=neighbors,timeWindow=timeWindow,MAX_DISTANCE=MAX_DISTANCE)
        gmlFiles = glob.glob(data_folder)
        Gall = nx.Graph()
        dfall = pd.DataFrame()
        
        for gfile in gmlFiles:
            print(gfile.split(os.sep)[-1])
            Gy = nx.read_gml(gfile)
            Ey = list(Gy.edges(data=True)) 
            Ny = list(Gy.nodes(data=True))
            
            Gall.add_nodes_from(Ny)   
            dfy = pd.DataFrame(Ey)
            dfall = pd.concat([dfall, dfy], ignore_index=True, sort=False)
            
        def funcConsolidateWeights(Gall, dfall):               
            dfall = dfall.groupby([dfall[0],dfall[1]]).agg({0: 'first',
                                                    1: 'first',
                                                    2: gc.func_sum })
        
            edges = [list(edge) for edge in list(dfall.values)]
            Gall.add_edges_from(edges)  
            return Gall
                   
        Gall = funcConsolidateWeights(Gall, dfall)
        GallU = Gall.to_undirected() 
        
        GallU = self._setGraphCommunityLabels(GallU,  
                             measure=la.RBConfigurationVertexPartition,
                             min_comm_size=5)
                
        filename = gc.getFileName(begin='net-ALL-years',neighbors=neighbors,
                               MAX_DISTANCE=MAX_DISTANCE, 
                               timeWindow=timeWindow, ext='gml',path=os.path.join(owd, 'net-All-Years'))
            
        figname = gc.getFileName(begin='net-ALL-years',neighbors=neighbors,
                               MAX_DISTANCE=MAX_DISTANCE, 
                               timeWindow=timeWindow, ext='png',path=os.path.join(owd, 'net-All-Years'))
        
        
        nx.write_gml(GallU, filename)
        # Remove self-loops
        self_loops = list(nx.selfloop_edges(GallU))
        GallU.remove_edges_from(self_loops)
        self.drawAllTimeCommunityGraph(GallU, figname, min_comm_size=5)       


    def getFolderFilesYears(self,folderData, pattern):
        
        data_folder = os.path.join(folderData, pattern)
        print('data_folder',data_folder)
        self.folderData = folderData
        ncFiles = glob.glob(data_folder)
        assert len(ncFiles) > 0, ("No files with the pattern {}".format(data_folder))
    
        dfiles = pd.DataFrame(ncFiles)
        dfiles.rename(columns={0:'FileAddress'}, inplace = True)
        return dfiles
    
    
    def printLinesGlobe(self, m):
        
        lats = m.drawparallels(np.arange(-90,90,30),labels=[1,1,0,1])
        lons = m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,60),labels=[1,1,0,1])
        # keys contain the plt.Line2D instances
        lat_lines = chain(*(tup[1][0] for tup in lats.items()))
        lon_lines = chain(*(tup[1][0] for tup in lons.items()))
        all_lines = chain(lat_lines, lon_lines)
        
        # cycle through these lines and set the desired style
        for line in all_lines:
            line.set(linestyle='-', alpha=0.3, color='w',linewidth=2)
            

    ## Prints the conventional white figure map using different colors for the components
    def drawAllTimeCommunityGraph(self, Gy, figname, min_comm_size=10, selection=None):
        
        lat = list(Gy.nodes(data = 'lat'))
        long = list(Gy.nodes(data = 'lng'))
        clabel = list(Gy.nodes(data = 'cLabel'))
        
        df = pd.DataFrame() 
        df['id'] = pd.DataFrame(lat)[0] # ID label in pos 0               
        df['lat'] = pd.DataFrame(lat)[1]
        df['lng'] = pd.DataFrame(long)[1]                
        df['degree'] = pd.DataFrame(Gy.degree)[1]
        df['community'] = pd.DataFrame(clabel)[1].astype(int)
        
        partition = df['community'].value_counts()        
        #removing the communities with less than 5 events over the years
        partition =  partition[partition >= min_comm_size]        
        if selection != None:
            df = df.loc[df['community'].isin(selection)] 
        else:
            df = df.loc[df['community'].isin(list(partition.index))]  
        
        pos = dict()
        for row in df.itertuples():
            pos[row.id] = [row.lng, row.lat]
          
        fig = plt.figure(figsize=(15, 15), edgecolor='w')     
        m = Basemap(resolution='c', 
                llcrnrlat=-60, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180, )
        m.drawcoastlines()
        m.drawcountries()   
        
        #nx.draw_networkx_edges(Gy, pos=pos, edge_color='blue', alpha=.5, width=1.0)
        
        m.scatter(df['lng'].values, df['lat'].values, latlon=True,
                    s=12.0, c= df['community'].values,
                  cmap='tab20', alpha=0.3, marker='o', edgecolor='k')
        
        # Adding community labels with arrows
        for community_label, group in df.groupby('community'):
            community_center = group[['lng', 'lat']].mean()
            offset = 25  # Ajuste a distância do rótulo em relação ao ponto
            if community_center['lng'] >= 0:
                plt.annotate(community_label, xy=(community_center['lng'], community_center['lat']), xytext=(offset, 0),
                             textcoords='offset points', ha='left', fontsize=11, color='blue',
                             arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle="->"))
            else:
                plt.annotate(community_label, xy=(community_center['lng'], community_center['lat']), xytext=(-offset, 0),
                             textcoords='offset points', ha='right', fontsize=11, color='blue',
                             arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle="->"))


        plt.savefig(figname, format='png', bbox_inches='tight')
        #plt.close()
        

    ## receive the gml file name and plot the graph
    def drawGraphMap(self,filename,figname,conf,comms,title=r'$\bf All \ period$', Community=True):
                
        Gy = nx.read_gml(filename)
        lat = list(Gy.nodes(data = 'lat'))
        long = list(Gy.nodes(data = 'lng'))
        df = pd.DataFrame() 
        df['id'] = pd.DataFrame(lat)[0]                
        df['lat'] = pd.DataFrame(lat)[1]
        df['lng'] = pd.DataFrame(long)[1]  
        df['degree'] = pd.DataFrame(Gy.degree)[1]
        print("MaxDegree ", df['degree'].max())
        
        comps = sorted(nx.connected_components(Gy), key = len, reverse=True)
        cid = 0
        for comp in comps:
            compl = list(comp)
            df.loc[df['id'].isin(compl),'comp'] = cid
            cid +=1
        
        pos = dict()
        for row in df.itertuples():
            pos[row.id] = [row.lng, row.lat]
          
        df.sort_values(by='degree',inplace=True,ascending=False)  
        df['Label'] = df['id'].astype(int)
        df['Community'] = conf.reindex(index= df['Label'].values,fill_value = '-1')['cLabel'].values
         
        fig = plt.figure(figsize=(10, 8), edgecolor='w')     
        m = Basemap(projection='eck4', lat_0=0, lon_0=0)
        m.drawcoastlines(linewidth=1,color='k')
        m.drawmapboundary(fill_color='white')
        
        # draw a shaded-relief image
        m.shadedrelief(scale=0.2,alpha=0.7)    
        self.printLinesGlobe(m)
        
        colorsComp = [comms[x.Community] if x.Community != '-1' else np.array([0.1, 0.5, 0.1, 1.])  for x in df.itertuples()] 
        nx.draw_networkx_edges(Gy,pos=pos,edge_color='blue', alpha=.9, width=2.0)
        
        if Community:
            m.scatter(df['lng'].values, df['lat'].values, latlon=True,
                        s=df['degree'].values*4, c=colorsComp, edgecolor='black',
                        alpha=0.5, marker='o')
        else:
         ## For creating the worldmaps with fire
            m.scatter(df['lng'].values, df['lat'].values, latlon=True,
                        s=df['degree'].values*4, c=np.log(df['degree'].values+1),
                      cmap='afmhot_r', alpha=0.6, marker='o')
        
        plt.title(title,fontsize = 18)
        folder = 'Community' if Community else 'Fire'
        imgPath = os.path.join(os.curdir, 'img'+folder)
        figname = os.path.join(imgPath, figname)
        
        plt.savefig(figname, format='png', bbox_inches='tight')
        #plt.close()
       
            
    def printGraph(self,gmlFiles,args):
        conf,comms,Community = args
        for gfile in gmlFiles:
            print(gfile)
            filename = os.path.basename(gfile).split('.gml')[0]     
            year = filename.split('_')[1]
            figname = 'fignet-'+filename+'.png'
            self.drawGraphMap(gfile,figname,conf,comms,year,Community)
        
        return []
           
        
    ## Function to generate the maps the overall year CodNet of war events.
    def printYearsNetworksfromGmlFiles(self,neighbors=3,timeWindow='year',
                                       MAX_DISTANCE=500, Community=True,
                                       graphsPath=os.path.join(owd, 'graphs')):
        
        data_folder = gc.getFilePattern(neighbors=neighbors,timeWindow=timeWindow,
                                        MAX_DISTANCE=MAX_DISTANCE,graphsPath=graphsPath)
        gmlFiles = list(glob.glob(data_folder))
        
        
        fileGML= gc.getFileName(begin='net-ALL-years',neighbors=neighbors,
                                       MAX_DISTANCE=MAX_DISTANCE, 
                                       timeWindow=timeWindow, path=os.path.join(owd, 'net-All-Years'))
            
        conf =  self.getCommunityLabel(fileGML)  
        
        conf.set_index('Label',drop=False,append=False,inplace=True)
        comms = conf['cLabel'].unique()
        #comms = np.sort(comms)
        colors = list(plt.cm.tab20b(np.linspace(0,1,20))) 
        colors.extend(list(plt.cm.tab20c(np.linspace(0,1,20))))
        if len(comms) > 40:
            colors.extend(list(plt.cm.gnuplot2_r(np.linspace(0,1,(len(comms)-40)))))
        
        
        comms = {str(i):colors[i] for i, _ in enumerate(colors)}
        folder = 'Community' if Community else 'Fire'
        imgPath = os.path.join(os.curdir, 'img'+folder)
        from pathlib import Path
        Path(imgPath).mkdir(parents=True, exist_ok=True)
    
        r = parallel.For(gmlFiles,self.printGraph,argsF=(conf,comms,Community)) 
        print(fileGML)
        filename = os.path.basename(fileGML).split('.gml')[0]     
        figname = 'fignet-'+filename+'.png'
        self.drawGraphMap(fileGML,figname,conf,comms,Community=Community)
      
      
    
    ## Function used to create the maps for each year of war.
    def print_AllYears_NetworksfromGmlFiles(self,neighbors=3,timeWindow='year',
                                       MAX_DISTANCE=500, Community=True,
                                       graphsPath=os.path.join(owd, 'net-All-Years')):
          
        fileGML= gc.getFileName(begin='net-ALL-years',neighbors=neighbors,
                                       MAX_DISTANCE=MAX_DISTANCE, 
                                       timeWindow=timeWindow, path=graphsPath)
            
        conf =  self.getCommunityLabel(fileGML)  
        
        conf.set_index('Label',drop=False,append=False,inplace=True)
        comms = conf['cLabel'].unique()
        colors = list(plt.cm.tab20b(np.linspace(0,1,20))) 
        colors.extend(list(plt.cm.tab20c(np.linspace(0,1,20))))
        
        if len(comms) > 40:
            colors.extend(list(plt.cm.gnuplot2_r(np.linspace(0,1,(len(comms)-40)))))    
        
        comms = {str(i):colors[i] for i, _ in enumerate(colors)}
                
        folder = 'Community' if Community else 'Fire'
        imgPath = os.path.join(graphsPath, 'img'+folder)
        from pathlib import Path
        Path(imgPath).mkdir(parents=True, exist_ok=True)
    
        print(fileGML)
        filename = os.path.basename(fileGML).split('.gml')[0]     
        figname = 'fignet-'+filename+'.png'
        
        self.drawGraphMap(fileGML,figname,conf,comms,Community=Community)
           

     

if __name__ == "__main__":
    
    teste = codnet()
    
    argv = sys.argv[1:]    
    try:
        opts, args = getopt.getopt(argv,"h:",["help"])
    except getopt.GetoptError:
        print ('program.py op=[year, month] MAX_DISTANCE=[100, 250, 500]')        
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ('program.py op=[year, month]')            
            sys.exit()
    
    timeWindow = str(argv[0])
    MAX_DISTANCE = int(argv[1])
        
    teste.printYearsNetworksfromGmlFiles(neighbors=11,timeWindow=timeWindow,MAX_DISTANCE=MAX_DISTANCE,Community=False)
    teste.printYearsNetworksfromGmlFiles(neighbors=11,timeWindow=timeWindow,MAX_DISTANCE=MAX_DISTANCE)
    
 