import time
import sys
import numpy as np
import networkx as nx
import os
from NSW import NSW
from HNSW_scan import SCAN
from tqdm import tqdm
from DataLoader import Loader
from HNSW_RyanLiGod import HNSW

dbname ='SIFTSMALL'  #数据集名字，'SIFT'  'SIFTSMALL'
k = 100  #搜索k最近邻，输入1-100#
dataPath="/home/like/data/"
print("Preparing dataset", dbname)
xt, xb, xq, gt = Loader(dbname).read(dataPath)
nb, d = xb.shape
nq, d = xq.shape

def add_two_dim_dict(thedict,keya,keyb,val):
    if keya in thedict:
        thedict[keya].update({keyb:val})
    else:
        thedict.update({keya:{keyb:val}})

def evaluate(mu, epsilon, m0, ef):
    layer_0_dict=dict()
    if os.path.exists(dbname+'_layer_0_dict'):
        print("使用上一次建的图")
        fr=open(dbname+'_layer_0_dict','r+')
        layer_0_dict=eval(fr.read())
        fr.close()
    else:
        print("从头建HNSW图，并保存level 0")
        hnsw = HNSW('l2', m0=m0, ef=ef)
        for ele in tqdm(xb, desc="NSW constructing..."):
            hnsw.add(ele,-1)
        for i in range(len(hnsw._graphs)):
            print("level",i,"points:",len(hnsw._graphs[i]))
        print("NSW-optimize-3 search")
        result=[]
        t0 = time.time()
        for i in range(len(xq)):
            idx = hnsw.search(xq[i], 1)
            result.append(idx)
        t1 = time.time()
        t=(t1 - t0) * 1000.0 / nq
        recalls = {}
        i = 1
        recalls[10]=-1
        while i <= k:
            recalls[i] = (result == gt[:, :1]).sum() / float(nq)
            i *= 10
        print("\t %7.3f ms per query, R@1 %.4f, R@10 %.4f" % (t, recalls[1],recalls[10]))

        layer = hnsw._graphs[0]
        for i in layer:
            for j in layer[i]:
                add_two_dim_dict(layer_0_dict,i,j,layer[i][j])
        fw=open(dbname+'_layer_0_dict','w+')
        fw.write(str(layer_0_dict))
        fw.close()


    print("统计邻边-调用SCAN算法")
    pairs = []
    for i in layer_0_dict:
        for j in layer_0_dict[i]:
            pairs.append([i,j])
    G = nx.Graph()
    print("共有%d条连边"%len(pairs))
    for line in pairs:
        source, target = line[0], line[1]
        G.add_edge(source, target)
    algorithm = SCAN(G, epsilon, mu)
    core_nodes, communities = algorithm.execute()
    coreNodes2Neighbors = algorithm.get_all_core_points()
    print("The number of core points: ", len(coreNodes2Neighbors))
    core_points = coreNodes2Neighbors.keys()
    hubs_outliers = algorithm.get_hubs_outliers(communities)
    print("core nodes: ", len(core_nodes))
    print('hubs: ', len(hubs_outliers[0]))
    print('outliers: ', len(hubs_outliers[1]))
    

    print("核心节点/桥节点 或者 核心节点&桥节点,建图")
    hnsw2  = NSW('l2', m0=m0, ef=ef)
    for i in tqdm(range(len(xb))):
        # if i in core_nodes or i in hubs_outliers[0]:
        # if i in core_points or i in hubs_outliers[0]:
        if i in core_points or i in hubs_outliers[1] or i in hubs_outliers[0]:
            hnsw2.add(xb[i],2)
        else:
            hnsw2.add(xb[i],1)
    for i in range(len(hnsw2._graphs)):
        print("level",i,"points:",len(hnsw2._graphs[i]))
    print("NSW-optimize-3 search")
    result2=[]
    t0 = time.time()
    for i in range(len(xq)):
        idx = hnsw2.search(xq[i], 1, 100, coreNodes2Neighbors)
        # [(5460, 137.42998), (5439, 138.89925)] when k = 2
        result2.append(idx)
    t1 = time.time()


    t=(t1 - t0) * 1000.0 / nq
    recalls2 = {}
    i = 1
    recalls2[10]=-1
    while i <= k:
        recalls2[i] = (result2 == gt[:, :1]).sum() / float(nq)
        i *= 10
    print("\t %7.3f ms per query, R@1 %.4f, R@10 %.4f" % (t, recalls2[1],recalls2[10]))


if __name__ == "__main__":
    # SCAN参数
    mu = 5
    epsilon = 0.3
    # HNSW参数
    m0 = 16
    ef = 128
    for _mu in [5,10,15,20,25]:
        evaluate(_mu, epsilon, m0, ef)
    for _epsilon in [0.15,0.2,0.25,0.3,0.35]:
        evaluate(mu, _epsilon, m0, ef)
    for _m0 in [8,16,24,32]:
        evaluate(mu, epsilon, _m0, ef)
    for _ef in [0.15,0.2,0.25,0.3,0.35]:
        evaluate(mu, epsilon, m0, _ef)