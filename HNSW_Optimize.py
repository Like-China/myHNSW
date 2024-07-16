import time
import sys
import numpy as np
# import faiss
import networkx as nx
import os

from HNSW_RyanLiGod import HNSW
from HNSW_scan import SCAN
from tqdm import tqdm
from DataLoader import Loader

# dbname =sys.argv[1]  #数据集名字，'SIFT'  'SIFTSMALL'
# k =int(sys.argv[2])  #搜索k最近邻，输入1-100#
# similarityLine=int(sys.argv[3])  #相似度百分比，输入1-100,
# todo =sys.argv[4]  #执行函数名，'HNSW-optimize-1'
dbname ='SIFTSMALL'  #数据集名字，'SIFT'  'SIFTSMALL'
k = 100  #搜索k最近邻，输入1-100#
similarityLine= 10  #相似度百分比，输入1-100,
todo = 'HNSW-optimize-1'  #执行函数名，'HNSW-optimize-1'
dataPath="/home/like/data/"
# SCAN参数
mu = 20
epsilon = 0.2
# HNSW参数
m0 = 16
ef = 128

    
    
print("Preparing dataset", dbname)
xt, xb, xq, gt = Loader(dbname).read(dataPath)
nb, d = xb.shape
nq, d = xq.shape

def add_two_dim_dict(thedict,keya,keyb,val):
    if keya in thedict:
        thedict[keya].update({keyb:val})
    else:
        thedict.update({keya:{keyb:val}})


if 'HNSW-faiss' == todo:
    print("HNSW-faiss begin")
    # Create an HNSW index
    index = faiss.IndexHNSWFlat(d, 32)
    #faiss.omp_set_num_threads(1)
    #index.hnsw.efConstruction = 40
    #index.hnsw.efSearch = 40
    #index = faiss.index_factory(d, "HNSW32,Flat")
    #>>> index.hnsw.nb_neighbors(1)
    #32
    #>>> index.hnsw.nb_neighbors(0)
    #64
    #ParameterSpace().set_index_parameter(index, "quantizer_efSearch")
    #ParameterSpace().set_index_parameters(index, "nprobe=128,quantizer_efSearch=512")

    # Add the vectors to the index
    index.add(xb)

    print("HNSW-faiss search")
    # Perform a search
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    t=(t1 - t0) * 1000.0 / nq
    recalls = {}
    recalls[10]=-1
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        print("R@",i,recalls[i])
        i *= 10
    print("\t %7.3f ms per query, R@1 %.4f" % (t, recalls[1]))

    print("Search with exact level-1")
    hnsw = index.hnsw
    # find level-1 elements (that are numbered as 2)
    levels = faiss.vector_to_array(hnsw.levels)
    level1, = np.where(levels == 2)
    nprobe = 10 # use 5 starting points per query vector
    Dc, Ic = faiss.knn(xq, xb[level1], nprobe)
    Ic = level1[Ic].astype('int32')
    D = np.zeros((nq, k), dtype='float32') 
    I = np.zeros((nq, k), dtype='int64')

    t0 = time.time()
    index.search_level_0(
        len(xq), faiss.swig_ptr(xq), 
        k, 
        faiss.swig_ptr(Ic), faiss.swig_ptr(Dc), 
        faiss.swig_ptr(D), faiss.swig_ptr(I), 
        nprobe    
    )
    t1 = time.time()
    t=(t1 - t0) * 1000.0 / nq
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        print("R@",i,recalls[i])
        i *= 10
    print("\t %7.3f ms per query, R@1 %.4f" % (t, recalls[1]))

if 'HNSW-random' == todo:
    print("HNSW-random begin")
    hnsw = HNSW('l2', m0=m0, ef=ef)
    for ele in tqdm(xb, desc="HNSW constructing..."):
        hnsw.add(ele,-2)
    for i in range(len(hnsw._graphs)):
        print("level",i,"points:",len(hnsw._graphs[i]))
    print("HNSW-random search")
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

if 'HNSW-optimize-1' ==todo:
    common_neighbors_number=dict()
    print("HNSW-optimize-1 begin")
    hnsw = HNSW('l2', m0=m0, ef=ef)
    for ele in tqdm(xb, desc="HNSW constructing..."):
        hnsw.add(ele,-1)
    for i in range(len(hnsw._graphs)):
        print("level",i,"points:",len(hnsw._graphs[i]))
    print("HNSW-optimize-1 search")
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

    print("计算相邻两个点的相似度")
    #先测试两层的效果
    #计算level 0， 任意相邻点的共同邻居点的数目占总邻居点的比例作为相似度，相似度高的只保留一个放在Level 1

    layer = hnsw._graphs[0]
    for i in range(len(layer)):
        point_1_neighbor_number=len(layer[i])
        tmp1=layer[i] # 存储的每个点到NN邻居的距离
        for j in layer[i]:
            # if i in common_neighbors_number:
            #     if j in common_neighbors_number[i]:
            #         continue
            point_2_neighbor_number=len(layer[j])
            similarity=0
            for z in layer[i].keys():
                if z in layer[j]:
                    similarity+=1
            
            similarity=similarity/(point_1_neighbor_number+point_2_neighbor_number)
            add_two_dim_dict(common_neighbors_number,i,j,similarity)
    print("相似度较高的选择一个点，放在Level0")
    choose={}
    for i in common_neighbors_number:
        for j in common_neighbors_number[i]:
            if i in choose or j in choose:
                continue
            if common_neighbors_number[i][j]>similarityLine/100.0:
                choose[i]=2
                choose[j]=1
    print('choose length:',len(choose))
    print("根据计算的点的Level，建图")
    hnsw2 = HNSW('l2', m0=m0, ef=ef)
    for i in tqdm(range(len(xb))):
        if i in choose and choose[i]==2:
            hnsw2.add(xb[i],2)
        else:
            hnsw2.add(xb[i],1)
    for i in range(len(hnsw2._graphs)):
        print("level",i,"points:",len(hnsw2._graphs[i]))
    print("HNSW-python search")
    result2=[]
    t0 = time.time()
    for i in range(len(xq)):
        idx = hnsw2.search(xq[i], 1)
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

if 'HNSW-optimize-2' ==todo:
    common_neighbors_number=dict()
    print("HNSW-optimize-2 begin")
    hnsw = HNSW('l2', m0=m0, ef=ef)
    for ele in tqdm(xb, desc="HNSW constructing"):
        hnsw.add(ele,-1)
    for i in range(len(hnsw._graphs)):
        print("level",i,"points:",len(hnsw._graphs[i]))
    print("HNSW-optimize-2 search")
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

    print("计算相邻两个点的相似度")
    #先测试两层的效果
    #计算level 0， 任意相邻点的共同邻居点的数目占总邻居点的比例作为相似度，相似度高的只保留一个放在Level 1

    layer = hnsw._graphs[0]
    for i in range(len(layer)):
        point_1_neighbor_number=len(layer[i])
        tmp1=layer[i]
        for j in layer[i]:
            if i in common_neighbors_number:
                if j in common_neighbors_number[i]:
                    continue
            point_2_neighbor_number=len(layer[j])
            similarity=0
            for z in layer[i].keys():
                if z in layer[j]:
                    similarity+=1
            
            similarity=similarity/(point_1_neighbor_number+point_2_neighbor_number)
            add_two_dim_dict(common_neighbors_number,i,j,similarity)
    print("相似度较高的选择一个点,放在Level0")
    choose={}
    for i in common_neighbors_number:
        for j in common_neighbors_number[i]:
            if (i in choose and choose[i]==1) or j in choose:
                continue
            if common_neighbors_number[i][j]>similarityLine/100.0:
                choose[i]=2
                choose[j]=1
    print('choose length:',len(choose))
    print("根据计算的点的Level,建图")
    hnsw2 = HNSW('l2', m0=m0, ef=ef)
    for i in tqdm(range(len(xb))):
        if i in choose and choose[i]==2:
            hnsw2.add(xb[i],2)
        else:
            hnsw2.add(xb[i],1)
    for i in range(len(hnsw2._graphs)):
        print("level",i,"points:",len(hnsw2._graphs[i]))
    print("HNSW-python search")
    result2=[]
    t0 = time.time()
    for i in range(len(xq)):
        idx = hnsw2.search(xq[i], 1)
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


if 'HNSW-optimize-3' ==todo:
    layer_0_dict=dict()
    if os.path.exists(dbname+'_layer_0_dict'):
        print("使用上一次建的图")
        fr=open(dbname+'_layer_0_dict','r+')
        layer_0_dict=eval(fr.read())
        fr.close()
    else:
        print("从头建HNSW图，并保存level 0")
        hnsw = HNSW('l2', m0=m0, ef=ef)
        for ele in tqdm(xb, desc="HNSW constructing..."):
            hnsw.add(ele,-1)
        for i in range(len(hnsw._graphs)):
            print("level",i,"points:",len(hnsw._graphs[i]))
        print("HNSW-optimize-3 search")
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
    core_points = algorithm.get_all_core_points()
    print(len(core_points))
    hubs_outliers = algorithm.get_hubs_outliers(communities)
    print("core nodes: ", len(core_nodes))
    print('hubs: ', len(hubs_outliers[0]))
    print('outliers: ', len(hubs_outliers[1]))
    

    print("核心节点/桥节点 或者 核心节点&桥节点,建图")
    hnsw2 = hnsw = HNSW('l2', m0=m0, ef=ef)
    for i in tqdm(range(len(xb))):
        # if i in core_nodes or i in hubs_outliers[0]:
        if i in hubs_outliers[0]:
            hnsw2.add(xb[i],2)
        else:
            hnsw2.add(xb[i],1)
    for i in range(len(hnsw2._graphs)):
        print("level",i,"points:",len(hnsw2._graphs[i]))
    print("HNSW-python search")
    result2=[]
    t0 = time.time()
    for i in range(len(xq)):
        idx = hnsw2.search(xq[i], 1)
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