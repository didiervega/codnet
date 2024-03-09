# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:50:41 2019

@author: Didier
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os, sys
from matplotlib.ticker import MultipleLocator
from itertools import chain
import graphConstruction as gc

import parallel
import codnet as coms

from mpl_toolkits.basemap import Basemap
sys.stdout.flush()
from scipy.stats import pearsonr, spearmanr

owd = os.getcwd()

class communityTS(coms.codnet):
    
    def __init__(self):          
        super().__init__()
        self.pattern='net-year_*'

    def printLinesGlobeSmall(self, m):
        lats = m.drawparallels(np.arange(-90,90,10),labels=[1,0,0,1])
        lons = m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,10),labels=[1,1,0,1])
        
        # keys contain the plt.Line2D instances
        lat_lines = chain(*(tup[1][0] for tup in lats.items()))
        lon_lines = chain(*(tup[1][0] for tup in lons.items()))
        all_lines = chain(lat_lines, lon_lines)
        
        # cycle through these lines and set the desired style
        for line in all_lines:
            line.set(linestyle='-', alpha=0.3, color='w',linewidth=2)

        
    def getValidConfNodes(self,dfPrec,dfTemp, conf):
        sP = set(dfPrec.index)
        sT = set(dfTemp.index)
        sC = set(conf.index)
        
        validIds = sP & sT & sC
        return conf.loc[validIds]            

    def initNewDataFrameYear(self,conf,year):
        
        dfComm = pd.DataFrame(conf[['Label','cLabel']]) 
        dfComm.set_index('Label',drop=False,append=False,inplace=True)
        dfComm.rename(columns={'Label':'gid'}, inplace=True)
        dfComm['n'] = np.nan
        dfComm['n_norm_comm'] = np.nan
        dfComm['m'] = np.nan
        dfComm['<k>'] = np.nan    
        dfComm['<s>'] = np.nan
        dfComm['lo'] = np.nan
        dfComm['tmp_i'] = np.nan
        dfComm['pre_i'] = np.nan
        dfComm['tmp_mean_community'] = np.nan
        dfComm['pre_mean_community'] = np.nan
        dfComm.sort_values(by='cLabel',inplace=True)
        dfComm.dropna(how='all',inplace=True)
        dfComm['year'] = year        
        return dfComm   
    
    
    #SAVE FOR ALL THE CELLS  ['n', 'm', '<k>', 'k_i', temp_i, prec_i] 
    def altFunc(self, dyears, args):
        
        dfPrec,dfTemp,conf,communities = args
        dfFull = pd.DataFrame() 
        for dyear in dyears:
            G = nx.read_gml(dyear)        
            year = int(dyear.split('_')[1].split('.')[0])
            dfComm = self.initNewDataFrameYear(conf, year)        
            precYear = dfPrec.loc[(dfPrec['year'] == year),['year','value']]
            tempYear = dfTemp.loc[(dfTemp['year'] == year),['year','value']]
                    
            for comm in communities.itertuples(): 
                        
                nodes = [x for x in conf.loc[(conf['cLabel'] == comm.cLabel),:].index]
                nodesC = [str(x) for x in nodes]                                                 
                H = G.subgraph(nodesC)
                n = nx.number_of_nodes(H)
                dfComm.loc[nodes,'n'] = n
                dfComm.loc[nodes,'n_norm_comm'] = n/comm.size
                dfComm.loc[nodes,'m'] = nx.number_of_edges(H)
                
                tempNodes = tempYear.reindex(index=nodes)
                tempNodes.dropna(axis=0,how='any',inplace=True)

                precNodes = precYear.reindex(index=nodes)
                precNodes.dropna(axis=0,how='any',inplace=True)                
                
                dfComm.loc[nodes,'tmp_mean_community'] = tempNodes['value'].mean()
                dfComm.loc[nodes,'pre_mean_community'] = precNodes['value'].mean()
                                
                if n == 0:
                    continue; 
                    
                strength = list(dict(H.degree(weight='weight')).values())
                degree = list(dict(H.degree).values())
                # <k>    
                avgK = 2*nx.number_of_edges(H)/n 
                dfComm.loc[nodes,'<k>'] = avgK
                
                #lo localization or complexity
                dfComm.loc[nodes,'lo'] = (self.complexity(degree) if avgK > 1.0 else 0.0)
                # <s>
                dfComm.loc[nodes,'<s>'] = np.mean(strength)
                 
                           
            precNodes = precYear.reindex(index=dfComm.index)
            tempNodes = tempYear.reindex(index=dfComm.index)
            #tmperature and precipitation
            dfComm.loc[precNodes.index,'pre_i'] = precNodes['value']            
            dfComm.loc[tempNodes.index,'tmp_i'] = tempNodes['value']
            
            dfFull = dfFull.append(dfComm, ignore_index=True, verify_integrity=False, sort=None)
        
        return dfFull.values.tolist()
    
    
    def _get_precipitation_AND_temperature_data(self):
        
        csvFile = os.path.join(gc.datafolder,'prec_dat.csv')
        dfPrec = pd.read_csv(csvFile, sep=',')
        dfPrec.dropna(axis=0,how='any',inplace=True)
        dfPrec.drop_duplicates(subset=['gid','year'],keep='first',inplace=True)
        dfPrec.set_index('gid',drop=False,append=False,inplace=True)
        dfPrec.rename(columns={'prec_gpcp':'value'}, inplace=True)
                
        csvFile = os.path.join(gc.datafolder,'gridtemp8914.csv')
        dfTemp = pd.read_csv(csvFile, sep=',')
        dfTemp.dropna(axis=0,how='any',inplace=True)
        dfTemp.drop_duplicates(subset=['gid','year'],keep='first',inplace=True)
        dfTemp.set_index('gid',drop=False,append=False,inplace=True)
        dfTemp.rename(columns={'temp':'value'}, inplace=True)
        return dfPrec, dfTemp
    
        
    def _construct_TableTSCells(self,timeWindow='year',MAX_DISTANCE=500,neighbors=3):

        opattern = gc.getFilePattern(neighbors=neighbors,timeWindow=timeWindow,MAX_DISTANCE=MAX_DISTANCE)
        fileGML= gc.getFileName(begin='net-ALL-years',neighbors=neighbors,
                                   MAX_DISTANCE=MAX_DISTANCE, 
                                   timeWindow=timeWindow, path=os.path.join(owd, 'net-All-Years'))
        
        print(opattern,'\n', fileGML)

        conf =  self.getCommunityLabel(fileGML)   
        conf.set_index('Label',drop=False,append=False,inplace=True) 
        dfFull = pd.DataFrame()     
        
        communities = pd.DataFrame(conf['cLabel'])        
        csize= communities.groupby('cLabel').agg({'cLabel': 'count'})    
        communities.drop_duplicates(inplace=True)
        communities.loc[:,'size'] = [csize.loc[cx,'cLabel'] for cx in communities['cLabel'].values]
        communities.sort_values(by=['size'],ascending=False,inplace=True)
    
        
        dfiles = self.getFolderFilesYears(self.folderData, opattern)
        #removing the last 3 years
        dfiles = dfiles[:-3]       
        
        dfPrec, dfTemp = self._get_precipitation_AND_temperature_data()
        conf = self.getValidConfNodes(dfPrec,dfTemp, conf)
        nodes = list(conf.index)
        
        dfPrec = dfPrec.loc[nodes,:]
        dfTemp = dfTemp.loc[nodes,:]
        lfiles = dfiles['FileAddress'].values.tolist()
                
        lFull = parallel.For(lfiles,self.altFunc,argsF=(dfPrec,dfTemp,conf,communities))   
        print('\n CoDNet network with ',len(lFull),' temporal nodes!')
        
        headers = ['gid','cLabel','n','n_norm_comm','m', '<k>', '<s>',
                   'lo','tmp_i','pre_i','tmp_mean_community','pre_mean_community','year']
    
        dfFull = pd.DataFrame(lFull, columns= headers)        
        return dfFull
       
    
    #SAVE FOR ALL THE CELLS  ['n', 'm', '<k>', 'be_i','k_i', temp_i, prec_i] 
    def _print_TableTSCells(self,timeWindow='year',MAX_DISTANCE=500,neighbors=3):   
        
        dfFull = self._construct_TableTSCells(timeWindow=timeWindow,MAX_DISTANCE=MAX_DISTANCE,neighbors=neighbors)
        dfFull.fillna(-9999, inplace=True)
        filename = gc.getFileName(begin='tableAllCells', timeWindow=timeWindow, neighbors=neighbors,
                                  MAX_DISTANCE=MAX_DISTANCE, ext='csv',path=os.curdir)
        
        dfFull.to_csv(filename,index=False,sep=',')
              
    
    def read_TableTSCells(self,timeWindow='year',MAX_DISTANCE=500,neighbors=3):
        
        filename = gc.getFileName(begin='tableAllCells', timeWindow=timeWindow, neighbors=neighbors,
                                  MAX_DISTANCE=MAX_DISTANCE, ext='csv',path=os.curdir)
        if os.path.exists(filename):
            with open(filename) as infile:
                dfFull = pd.read_csv(infile)
        else:
            dfFull = self._print_TableTSCells(timeWindow=timeWindow,MAX_DISTANCE=MAX_DISTANCE,neighbors=neighbors)    
        
        return dfFull
        
        
    #cols = # ['n',"n_norm_comm","m","<k>","<s>","lo",'tmp_i','pre_i','tmp_mean_community','pre_mean_community'] 
    #cols =  ['n',"m"]    
    def drawTs(self, cols, timeWindow='year',MAX_DISTANCE=500,neighbors=3):
       
        pltname = "TS-"
        for co in cols:
            pltname = pltname + co + "-"
        pltname = pltname.replace('<','_').replace('>',"")
             
        pltname = gc.getFileName(begin=pltname, timeWindow=timeWindow, neighbors=neighbors,
                                  MAX_DISTANCE=MAX_DISTANCE, ext='png',path=os.curdir)
        
        dfFull = self.read_TableTSCells(timeWindow=timeWindow,MAX_DISTANCE=MAX_DISTANCE,neighbors=neighbors)  
        lcorr = list()
        dfFull.dropna(inplace=True)
        dfFull = dfFull.loc[~((dfFull['lo'] == -9999) | (dfFull['tmp_i'] == -9999))]                
        dfComm = dfFull.groupby(by=['cLabel','year']).mean()
        dfComm.reset_index(inplace=True)
        dfComm = dfComm.groupby(by=['cLabel'])
        
        rows = np.sort(dfFull['cLabel'].unique())
        nrows= len(dfComm)
        ncols = len(cols)+2
        
        nplot = 1
        plt.rcParams['figure.figsize'] = (7*ncols, 1.9*nrows)
        figP = plt.figure()
        
        for i in rows:
            row = dfComm.get_group(i) 
            
            if len(row) < 11:
                continue
            
            xdata = row['year'].values
            tempR = row['tmp_mean_community'].values
            precR = row['pre_mean_community'].values
            
            for col in cols:
                plt.subplot(nrows, ncols, nplot)
                nplot+=1
                data = row[col].values                
                plt.plot(xdata, data, 'o-', label="cluster "+str(row['cLabel'].values[0])+' '+col)
                plt.legend(loc="best")
                ax = plt.gca()
                plt.xticks(np.arange(1989, 2014, 2), fontsize=8)
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ax.xaxis.grid(True)
                
                rT, rpvT = pearsonr(data, tempR)    # Pearson's r
                sT, spvT = spearmanr(data, tempR)    # Pearson's r
                sP, spvP = spearmanr(data, precR) 
                rP, rpvP = pearsonr(data, precR) 
                
                lcorr.append({"cLabel":row['cLabel'].values[0],"size":len(row),
                              "timeWindow":timeWindow,
                              "Distance":MAX_DISTANCE,"neighbors":neighbors,
                              "measure":col, "rT":rT, 'rpvT':rpvT,'sT':sT,'spvT':spvT,
                              'rP':rP,'rpvP':rpvP,'sP':sP,'spvP':spvP})
                
                
            plt.subplot(nrows, ncols, nplot)
            nplot+=1
            plt.plot(xdata, tempR, 'o-', label='MeanTemp', color='red')
            plt.legend(loc="best")
            ax = plt.gca()
            plt.xticks(np.arange(1989, 2014, 2), fontsize=8)
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.xaxis.grid(True)
            
            plt.subplot(nrows, ncols, nplot)
            nplot+=1
            plt.plot(xdata, precR,'o-', label='MeanPrec', color='orange')
            plt.legend(loc="best")
            ax = plt.gca()
            plt.xticks(np.arange(1989, 2014, 2), fontsize=8)
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.xaxis.grid(True)

        plt.subplots_adjust(wspace=0.13, hspace = 0.18)
        plt.savefig(pltname, format='png', bbox_inches='tight')
        return pd.DataFrame(lcorr)
    
 

    def draw_TSMap(self, corrDf, corr_measure="r", climate_measure='T'):
         
        timeWindow = corrDf['timeWindow'].values[0]
        MAX_DISTANCE = corrDf['Distance'].values[0]
        neighbors = corrDf['neighbors'].values[0]
        
        corr_m = corr_measure + climate_measure
        corr_pv = corr_measure + 'pv' + climate_measure
        
        fileGML= gc.getFileName(begin='net-ALL-years',neighbors=neighbors,
                                       MAX_DISTANCE=MAX_DISTANCE, 
                                       timeWindow=timeWindow, path=os.path.join(owd, 'net-All-Years'))
        Gy = nx.read_gml(fileGML)
        locations = pd.DataFrame(dict(Gy.nodes(data=True))).T
        del locations['month'], locations['year'], locations['event']
        locations.drop_duplicates(inplace=True) 
        locations.index = locations.index.astype(int)
        
        dfFull = self.read_TableTSCells(timeWindow=timeWindow,MAX_DISTANCE=MAX_DISTANCE,neighbors=neighbors) 
        dfFull = dfFull[['gid','cLabel']].drop_duplicates()
        cols = corrDf['measure'].unique()
        nrows= 1
        ncols = 3
        
        nplot = 1
        plt.rcParams['figure.figsize'] = (10*ncols, 7*nrows)
        figP = plt.figure()    
        dfComm = dfFull.groupby(by='cLabel')  
        
        corr_name = "Pearson" if corr_measure == "r" else "Spearman"
        
        print(f"{corr_name} correlation with all-period Codnet",end=" ")
        print(f"constructed with time window {timeWindow}, max.distance",end=" ")
        print(f"{MAX_DISTANCE}, neighbors {neighbors}\nSignificant",end=" ")
        print(f"communities (p-value < 0.05) with {climate_measure}:")
        
        for col in cols:              
            nodesDf = pd.DataFrame()
            corrCol = corrDf.loc[(corrDf['measure']==col) & (corrDf[corr_pv] <= 0.05)]
            comms = list(corrCol['cLabel'].unique())
            print(f"\t Communties: {comms} for metric {col}")
            
            for i in comms:
                nodes = dfComm.get_group(i)['gid'].values 
                nDf = locations.reindex(index= nodes).dropna()
                
                nDf['corr'] = corrCol.loc[corrCol['cLabel'] == i,corr_m].values[0]
                #nDf['cLabel'] = i
                nodesDf= nodesDf.append(nDf,ignore_index=False)
                
            plt.subplot(nrows, ncols, nplot)
            nplot+=1 
                    
            m = Basemap(projection='eck4', lat_0=0, lon_0=0)
            m.drawcoastlines(linewidth=1,color='k')
            m.drawcountries(linewidth=1,color='grey')
            m.drawmapboundary(fill_color='white')        
            # draw a shaded-relief image
            m.shadedrelief(scale=0.2,alpha=0.7)  
            self.printLinesGlobe(m)
            
            
            if comms != []:
                title='correlation of {} with {}'.format(col, corr_m)             
                m.scatter(nodesDf['lng'].values, nodesDf['lat'].values, latlon=True,
                s= 10.8, c= nodesDf['corr'].values,
                  cmap='seismic_r', alpha=0.6, marker='o') #edgecolor='white'
                
            else:
                m.scatter([0.0], [0.0], latlon=True,
                s= 10.8, c= [0.0],
                  cmap='seismic_r', alpha=0.6, marker='o') #edgecolor='white'
                title='NO significant p-vaulue with {}'.format(col) 
            
            plt.clim(-1, 1)
            plt.title(title,fontsize = 18)
            plt.colorbar(label='correlation', shrink=0.37)    

        print("\n")
        plt.subplots_adjust(wspace=-0.02, hspace = -0.4) 
        figname= gc.getFileName(begin='corrGraph-'+corr_m,neighbors=neighbors,
                                        MAX_DISTANCE=MAX_DISTANCE,ext='png', 
                                        timeWindow=timeWindow, 
                                        path=os.path.join(owd, 'net-All-Years'))
       
        plt.savefig(figname, format='png', bbox_inches='tight')
            
    
    def draw_TSMapComunity(self, cols, timeWindow ='year', MAX_DISTANCE=500,neighbors=3, communities=[11,27,37,24]):
         
        conf = gc.getConfEventData()
        conf.set_index('gid',drop=False,append=False,inplace=True)
        
        pltname = "TS_SelComm-"
        for co in cols:
            pltname = pltname + co + "-"
        pltname = pltname.replace('<','_').replace('>',"")
             
        pltname = gc.getFileName(begin=pltname, timeWindow=timeWindow, neighbors=neighbors,
                                  MAX_DISTANCE=MAX_DISTANCE, ext='png',path=os.curdir)
        
        dfFull = self.read_TableTSCells(timeWindow=timeWindow,MAX_DISTANCE=MAX_DISTANCE,neighbors=neighbors)    
        dfFull.dropna(inplace=True)
        dfFull = dfFull.loc[~((dfFull['lo'] == -9999) | (dfFull['tmp_i'] == -9999))]                
        dfComm = dfFull.groupby(by=['cLabel','year']).mean()
        dfComm.reset_index(inplace=True)
        dfComm = dfComm.groupby(by=['cLabel'])
               
        rows = np.sort(dfFull['cLabel'].unique())
        assert set(communities) & set(rows) == set(communities) , 'wrong community names'
        rows = np.array(communities)
        
        nrows= len(rows)
        ncols = len(cols) + 1
     
        plt.rcParams['figure.figsize'] = (10*ncols, 5*nrows)
        figP = plt.figure()
            
        measureLabel = {'<k>':r' $Average$ $intensity$',
                        'n':r' $Dispersion$', 
                        'm': r" $Intraconnectivity$"}
        
        def plotLine(nplot, data, xdata, label):
            plt.subplot(nrows, ncols, nplot) 
            plt.plot(xdata, data, 'o-', label=label)
            #https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
            plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0,fontsize=20)
            #plt.legend(loc="best",fontsize=15)
            ax = plt.gca()
            plt.xticks(np.arange(1989, 2014, 2), fontsize=13)
            plt.yticks(fontsize=13)
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.xaxis.grid(True)

        nplot = 0        
        count = 1  
        lit = 4
        initR = 1
        for i in rows:        
            commIds = dfFull.loc[(dfFull['cLabel'] == i),'gid'].unique()   
            commCount = conf.loc[commIds]      
            commCount.sort_values(by='year',inplace=True)
            commCount['stateSum'] = commCount['state']
            commCount['stateMean'] = commCount['state']
            commCount = commCount.loc[commCount['year'] <= 2014]
            commCountYear = commCount.groupby(by=['year']).agg({'stateSum':'sum','stateMean':'mean'})
        
            row = dfComm.get_group(i) 
            xdata = row['year'].values            
            for col in cols:
                data = row[col].values 
                label= r"${ \bf ("+str(count)+")}$ Community "+str(row['cLabel'].values[0])+", "+measureLabel[col]
                nplot+=1
                plotLine(nplot, data, xdata, label)  
                initR+=4    
                count+=1 
                
                if count == lit:
                    xticks = commCountYear.index
                    data = commCountYear['stateSum'].values#                     
                    label=r"${ \bf ("+str(count)+")}$ Community "+str(row['cLabel'].values[0]) +r", $Conflicts$  $Sum$ "
                    nplot+=1
                    plotLine(nplot, data, xticks, label)                
                    count+=1 
                    lit +=4
                
        plt.subplots_adjust(wspace=0.13, hspace = 0.28)
        plt.savefig(pltname, format='png', bbox_inches='tight')
     
  
if __name__ == "__main__":
    
    teste = communityTS()   
    #cols = # ['n',"n_norm_comm","m","<k>","<s>","lo",'tmp_i','pre_i','tmp_mean_community','pre_mean_community'] 
    cols =  ["<k>",'n', "m"]
          
    
    neighbors = [3]
    MAX_DISTANCE = [500] # km[100, ]
    timeWindow = ['month'] 
    
    teste.draw_TSMapComunity(cols, timeWindow='month', MAX_DISTANCE=500, neighbors=3, communities=[11,27,37,24])
    
    
    for k in neighbors:
        for distance in MAX_DISTANCE:
            for twindow in timeWindow:
                teste.read_TableTSCells(timeWindow=twindow,MAX_DISTANCE=distance,neighbors=k)
                corrDf = teste.drawTs(cols, timeWindow=twindow,MAX_DISTANCE=distance,neighbors=k)
                
                teste.draw_TSMap(corrDf, corr_measure="r", climate_measure='T')
                teste.draw_TSMap(corrDf, corr_measure="s", climate_measure='T')
                teste.draw_TSMap(corrDf, corr_measure="r", climate_measure='P')
                teste.draw_TSMap(corrDf, corr_measure="s", climate_measure='P')
   
                
          