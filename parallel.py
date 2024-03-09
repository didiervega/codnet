# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:41:28 2020

@author: Didier Vega-Oliveros
"""

import multiprocessing as mp
import math


def worker_job(lock, listFeat, task_queue, result_list, function, argsF):
       
    proc_name = mp.current_process().name
             
    while True:
        try:
            next_task = task_queue.get_nowait()
        
        except:
            print(str(proc_name)+': Exiting- Queue empty! ',end="") 
            break
        else:            
            idxFeat = next_task
            
            if len(idxFeat) <= 0:
                print('-------------------------------------------------')
                continue
            
            lFeat = [listFeat[x] for x in idxFeat]
            print(str(proc_name)+' : '+str(idxFeat[0])+'-'+str(idxFeat[-1]))            
            resps = function(lFeat, argsF)
            if(len(resps) > 0):               
                with lock: 
                    result_list.extend(resps)

    return True




def For(listFeat, function, argsF):
          
    num_workers = mp.cpu_count()
    numPJob = math.ceil(len(listFeat)/float(num_workers))
    
    mgr = mp.Manager()
    task_queue = mgr.Queue()   
    result_list = mgr.list() 
    lock = mgr.Lock()
       
    idxBL = list()
    for idxB in range(len(listFeat)):        
        idxBL.append(idxB)
        if len(idxBL) >= numPJob:
            task_queue.put(idxBL)
            idxBL = list()
       
    if len(idxBL) > 0:            
        task_queue.put(idxBL)
    
    # Start consumers
    processes = []
    
    print ('Creating '+str(num_workers)+ ' workers processing ',numPJob,' jobs')  
    for w in range(num_workers):
        p = mp.Process(target=worker_job, args=(lock, listFeat, task_queue, result_list, function, argsF))
        processes.append(p)
        p.start()
        
    for w in processes:
        w.join()
    
    return list(result_list) 
    
    
    

if __name__ == "__main__":

    print('main') 
    
   