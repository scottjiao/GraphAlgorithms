# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:56:08 2018

@author: ziang
"""
#read me:
'''


*******************************************************************************
已将各种算法封装到pj类之中，详见类的方法的说明

注：所有算法均手写，只有文件的读取过程调用了snap库，以及kmeans聚类过程调用了sklearn库

可视化采用pyechart的graph类

若想阅读代码使用示例，请拉到最下

请先直接F5运行整个文件后再在IDE中调试，否则第36行会报错（__file__不存在）
*******************************************************************************



'''

import sys
import snap
import matplotlib.pyplot as plt
import numpy as np
from pyecharts import Graph
import random
from sklearn.cluster import KMeans
from copy import deepcopy
import os
path=os.path.realpath(os.path.dirname(__file__))
os.chdir(path)


#求给定社区划分的模块度
def modularity(nodes,edges,community):
    #nodes : a list
    #edges : a adjacency matrix
    #community : a dict whose key=node,value=order of community
    edges=np.array(edges)
    num_com=max(community.keys())
    m=sum([sum(x) for x in edges])/2
    S=[]
    for n in nodes:
        line=[]
        for com in range(num_com):
            
            if com ==community[n]:
                line.append(1)
            else:
                line.append(0)
        S.append(line)
    S=np.array(S)
    deg=[sum(x) for x in edges]
    B=np.zeros((len(nodes),len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            B[i][j]=edges[i][j]-deg[i]*deg[j]/(2*m)
    Q=np.trace(np.dot(np.dot(S.T,B),S))/(2*m)
    return Q

def topN(PRankH, n):
    nodes = [w for w in PRankH]
    Top = sorted(nodes, key=lambda item: PRankH[item], reverse = True)
    return Top[:n]

    
    #封装所有函数和算法
class pj():
    def __init__(self):
        self.GraphName=None
        self.ComFileName=None
        self.SnapGraph=None
        self.ComStructDict={}
        self.NeighborDict=None
        self.nodes=None
        self.edgeMatrix=None
        self.NeighborDict_filter=None
        self.categories={}
        self.D=None
    def load(self,GraphName,ComFileName=None):
        #载入图
        self.GraphName= GraphName
        G = snap.LoadEdgeList(snap.PUNGraph,self.GraphName, 0, 1)
        self.SnapGraph=G
        
        
        #载入社区
        self.ComFileName=ComFileName
        if ComFileName:
            comf=open(self.ComFileName,'r')
            com0=comf.readlines()
            com=dict([(int(x.split()[0]),int(x.split()[1])) for x in com0])
            categories=com0
            categories=list(set([x.split()[1] for x in categories]))
            self.categories['groundtruth']=categories
            self.ComStructDict['groundtruth']=com
            comf.close()
        
    def NeighborDict_generate(self):
        #生成邻居索引字典
        m={}
        for n in self.SnapGraph.Nodes():
            idn=n.GetId()
            m[idn]=[]
            for j in n.GetOutEdges():
                if j!=idn:
                    m[idn].append(j)
            m[idn]=m[idn]
        self.NeighborDict=m
    def Filter_dict(self):
        #过滤，认为存在双向连接的才算好友关系
        m_filter={}
        for n in self.NeighborDict:
            temp=[]
            for j in self.NeighborDict[n]:
                if n in self.NeighborDict[j]:
                    temp.append(j)
            m_filter[n]=temp
        self.NeighborDict_filter=m_filter
        
        
    
        
    def degDist(self):
        # 度分布 
        DegToCntV = snap.TIntPrV()
        snap.GetDegCnt(self.SnapGraph, DegToCntV)
        degree = []
        counts = []
        for item in DegToCntV:
            counts.append(item.GetVal2())
            degree.append(item.GetVal1())
        plt.title('degree ditribution')
        plt.xlabel('degree')
        plt.ylabel('frequency')
        plt.plot(degree[0:50], counts[0:50])
        plt.show()

    def centralNodes(self,n):
        # 中心节点
        PRankH = snap.TIntFltH()
        snap.GetPageRank(self.SnapGraph, PRankH)
        self.topnlist=topN(PRankH, n)

    def balanced_kmeans(self,num_com=5,n_clusters=10):
        #计算特征向量，进行K-means聚类得到均衡社区
        #在此过程中计算度矩阵
        D=np.zeros(self.edgeMatrix.shape)
        for i in range(self.edgeMatrix.shape[0]):
            D[i][i]=sum(self.edgeMatrix[i,:])
        self.D=D
        laplacian=D-self.edgeMatrix
        eigVa,eigVe=np.linalg.eig(laplacian)
        eigVe=np.array([[float(y) for y in x] for x in eigVe])[:,1:num_com]
        eigVa=np.array([float(y) for y in eigVa])[1:num_com]
        
        estimator = KMeans(n_clusters=n_clusters)#构造聚类器
        estimator.fit(eigVe)#聚类
        label_pred = estimator.labels_ #获取聚类标签
        
        com=dict([(int(x),int(label_pred[x])) for x in range(len(label_pred))])
        categories=list(set(label_pred))
        self.categories['balanced']=categories
        self.ComStructDict['balanced']=com
        print 'm is :'
        print modularity(self.nodes,self.edgeMatrix,self.ComStructDict['balanced'])
        return estimator
        
    def visualize(self,rangee,com_name):
        #可视化为名为render的html文件
        nodes=[]
        for n in self.SnapGraph.Nodes():
            if n.GetId() in rangee:
                size=np.sqrt(n.GetDeg())
                if n.GetId() in self.topnlist:
                    size=20
                nodes.append({"name":n.GetId(),
                              "symbolSize":size,
                              "category":self.ComStructDict[com_name][n.GetId()]})
                
        links=[]
        for n in self.SnapGraph.Nodes():
            if n.GetId() in rangee:
                for j in n.GetOutEdges():
                    if j in rangee:
                        links.append({"source": n.GetId(), "target": j})
        
        graph = Graph("try")
        graph.add("", nodes, links,categories=self.categories[com_name], repulsion=50,line_curve=0.2,
                  graph_edge_symbol= ['circle', 'arrow'])
        graph.render()
        
        
        
    def getEdgeMatrix(self):
        #基于邻居字典构造邻接矩阵：

        edges=np.zeros((len(self.NeighborDict),len(self.NeighborDict)))
        for i in range(len(self.NeighborDict)):
            for j in range(len(self.NeighborDict)):
                if i!=j:
                    if j in self.NeighborDict[i]:
                        edges[i][j]=1
                        edges[j][i]=1
        nodes=list(range(len(self.NeighborDict)))
        self.nodes=nodes
        self.edgeMatrix=edges
    
    def walkTrap(self,t=3):
    #实现随机游走算法，t为步长
        A=self.edgeMatrix
        D=np.zeros(A.shape)
        for i in range(A.shape[0]):
            if self.D[i][i]!=0:
                D[i][i]=1/self.D[i][i]
        print 1
        
        P=np.dot(D,A)
        Pt=P
        for i in range(t-1):
            Pt=np.dot(Pt,P)
        r=np.zeros(D.shape)
        for i in range(D.shape[0]):
            print 'i:'
            print i
            for j in range(D.shape[0]):
                r[i][j]=np.sqrt(np.sum([(Pt[i][k]-Pt[j][k])**2/D[k][k] for k in range(D.shape[0]) if D[k][k]!=0]))
        #要跑20min
        #开始社区分类
        print 2

        Qm=0
        com_iter=dict([(i,[i]) for i in range(D.shape[0])])
        while True:
            print 'len(com_iter) is now'
            print len(com_iter)
            if len(com_iter)<=1:
                break
            temp=np.zeros((len(com_iter),len(com_iter)))
            #待优化
            for i in range(len(com_iter)):
                for j in range(len(com_iter)):
                    temp[i][j]=np.sqrt(
                        np.sum(
                        [
                            (np.average([Pt[x][k] for x in com_iter[i]])-np.average([Pt[y][k] for y in com_iter[j]]))**2/D[k][k]   \
                        for k in range(D.shape[0]) if D[k][k]!=0
                        ]
                    )
                    )
            chosen=np.where( temp == np.max(temp))
            chosen_index=int(np.min(chosen))
            drop_index=int(np.max(chosen))
            com_iter[chosen_index]=com_iter[chosen_index]+com_iter[drop_index]
            del com_iter[drop_index]
            #转社区字典:
            com_dict={}
            for key in com_iter:
                for x in com_iter[key]:
                    com_dict[x]=key
            qm=modularity(self.nodes,self.edgeMatrix,com_dict)
            if qm>Qm:
                Qm=qm
                MaxQ_com=deepcopy(com_dict)
       
        index_dict={}
        for item in enumerate(list(set(MaxQ_com.values()))):
            index_dict[item[1]]=item[0]
        self.ComStructDict['Random Walk']={}
        for x in MaxQ_com:
            self.ComStructDict['Random Walk'][x]=index_dict[MaxQ_com[x]]
        self.categories['Random Walk']=list(set(self.ComStructDict['Random Walk'].values()))
        print 'm is :'
        print modularity(self.nodes,self.edgeMatrix,self.ComStructDict['Random Walk'])


    def labelPropagation(self,thresh=10):
        n=len(self.nodes)
        label_iter=dict([(i,i) for i in range(n)])
        self.NeighborDict
        iters=0
        while True:
            print 'iter'
            print iters
            for x in label_iter:
                temp={}
                for neighbor in self.NeighborDict[x]:
                    if label_iter[neighbor] in temp:
                        temp[label_iter[neighbor]]+=1
                    else:
                        temp[label_iter[neighbor]]=1
                if temp=={}:
                    label_iter[x]='isolated'  
                else:
                    max_label_count=0
                    max_labels=[label_iter[x]]
                    for label in temp:
                        if temp[label]>max_label_count:
                            max_label_count=temp[label]
                            max_labels=[label]
                        elif temp[label]==max_label_count:
                            max_labels.append(label)
                    max_label=random.sample(max_labels,1)
                    label_iter[x]=max_label[0]
            iters+=1
            if len(set(label_iter.values()))<=thresh:
                break
        index_dict={}
        for item in enumerate(list(set(label_iter.values()))):
            index_dict[item[1]]=item[0]
        self.ComStructDict['Propagation']={}
        for x in label_iter:
            self.ComStructDict['Propagation'][x]=index_dict[label_iter[x]]
        self.categories['Propagation']=list(set(self.ComStructDict['Propagation'].values()))
        print 'Propagation m is :'
        print modularity(self.nodes,self.edgeMatrix,self.ComStructDict['Propagation'])
        
        

    def cliqueFilter(self,K):
        #找到团集
        if K==3:
            V=[]
            for i in range(1005):
                for j in self.NeighborDict[i]:
                    for k in self.NeighborDict[j]:
                        if i in self.NeighborDict[k]:
                            if i<j and j<k:
                                V.append((i,j,k))
                                
            #找到诱导图的边集
            mr=[]
            dic={}
            edge={}
            for v in V:
                mr.append(((v[0],v[1]),v))
                mr.append(((v[0],v[2]),v))
                mr.append(((v[1],v[2]),v))
            for item in mr:
                if item[0] in dic:
                    dic[item[0]].add(item[1])
                else:
                    dic[item[0]]=[item[1]]
                dic[item[0]]=set(dic[item[0]])
            for v in V:
                edge[v]=set()
                for i in dic[(v[0],v[1])]:
                    edge[v].add(i)
                for i in dic[(v[0],v[2])]:
                    edge[v].add(i)
                for i in dic[(v[1],v[2])]:
                    edge[v].add(i)
                edge[v]=set(edge[v])
                edge[v].remove(v)
        if K==4:
            #找到团集
            V=[]
            for i in range(1005):
                for j in self.NeighborDict[i]:
                    if i<j:
                        for k in self.NeighborDict[j]:
                            if j<k:
                                if i in self.NeighborDict[k]:
                                    for l in self.NeighborDict[k]:
                                        if k<l:
                                            if l in self.NeighborDict[i] and l in self.NeighborDict[j]:
                                                V.append((i,j,k,l))
                                    
            #建立边集：
            mr=[]
            dic={}
            edge={}
            for v in V:
                mr.append(((v[0],v[1],v[2]),v))
                mr.append(((v[0],v[2],v[3]),v))
                mr.append(((v[1],v[2],v[3]),v))
                mr.append(((v[0],v[1],v[3]),v))
            for item in mr:
                if item[0] in dic:
                    dic[item[0]].append(item[1])
                else:
                    dic[item[0]]=[item[1]]
            for v in V:
                edge[v]=[]
                for i in dic[(v[0],v[1],v[2])]:
                    edge[v].append(i)
                for i in dic[(v[0],v[2],v[3])]:
                    edge[v].append(i)
                for i in dic[(v[1],v[2],v[3])]:
                    edge[v].append(i)
                for i in dic[(v[0],v[1],v[3])]:
                    edge[v].append(i)
            
                edge[v]=list(set(edge[v]))
                edge[v].remove(v)
        #寻找连通分支
        explored=set()
        queue=[x for x in V]
        com_dict={}
        stack=[]
        initialcom=1
        t=0
        while True:
            if queue==[]:
                break
            seed=queue.pop(0)
            stack.append(seed)
            
            
            
            while True:
                if t%1000 ==1:
                    print len(explored)
                    print initialcom
                if stack==[]:
                    initialcom+=1
                    break
                temp=stack.pop()
                t+=1
                explored.add(temp)
                if temp in queue:
                    queue.remove(temp)
                com_dict[temp]=initialcom
                for x in edge[temp]:
                    if x not in explored and x not in stack:
                        stack.append(x)
                
        self.cliqueComDict=com_dict
        for x in com_dict:
            if com_dict[x]>1:
                print x,com_dict[x]        

    #为 link prediction的一个embedding
    def WL_algorithm(self,two_nodes_ids,N=10,NeighborDict=None):
        #input:(node1,node2)
        #output: a vector decoded by WL_algorithm
        if NeighborDict==None:
            NeighborDict=self.NeighborDict
        node_label={two_nodes_ids[0]:1,two_nodes_ids[1]:1}
        i=1
        enough=False
        while True:
            if len(node_label)>N:
                enough=True
                break
            if i>5 :
                break
            i+=1
            for node in NeighborDict[two_nodes_ids[0]]:
                if node not in node_label:
                    node_label[node]=i
            for node in NeighborDict[two_nodes_ids[1]]:
                if node not in node_label:
                    node_label[node]=i
        
        little_neighbor_dict={}
        for x in node_label:
            little_neighbor_dict[x]=list(set(NeighborDict[x]).intersection(set(node_label.keys())))
        #排序
        i=0
        while True:
            if i >10:
                if enough:
                    rank_list=sorted([(node_label[x],x) for x in node_label])[0:N]
                    order=[x[1] for x in rank_list]
                    break
                else:
                    rank_list=sorted([(node_label[x],x) for x in node_label])
                    order=[x[1] for x in rank_list]
                    break
            neighbor_dict={}
            for x in node_label:
                neighbor_dict[x]=[]
                for y in little_neighbor_dict[x]:
                    neighbor_dict[x].append(node_label[y])
                neighbor_dict[x]=sum(neighbor_dict[x])
            rank_list=sorted(list(set(neighbor_dict.values())))
            label_rank_dict={}
            for j in range(len(rank_list)):
                label_rank_dict[rank_list[j]]=j
                
            #print label_rank_dict
            for x in node_label:
                node_label[x]=label_rank_dict[neighbor_dict[x]]
            i+=1
        #构造embedding向量
        edges=[]
        if enough:
            for i in range(N-1):
                if i!=0:
                    for j in range(N-i-1):
                        if order[i] in little_neighbor_dict[order[j+i+1]]:
                            edges.append(1)
                        else:
                            range_list=[order[j+i+1]]
                            dk=0
                            for d in range(N):
                                if order[i] in range_list:
                                    dk=1./(d+1)
                                    break
                                for x in range_list:
                                    range_list=list(set(range_list+list(set(order).intersection(set(little_neighbor_dict[x])))))
                            edges.append(dk)
                elif i==0:
                    for j in range(N-i-2):
                        if order[i] in little_neighbor_dict[order[j+i+2]]:
                            edges.append(1)
                        else:
                            range_list=[order[j+i+2]]
                            dk=0
                            for d in range(N):
                                if order[i] in range_list:
                                    dk=1./(d+1)
                                    break
                                for x in range_list:
                                    range_list=list(set(range_list+list(set(order).intersection(set(little_neighbor_dict[x])))))
                            edges.append(dk)
        else:
            length=len(order)
            for i in range(N-1):
                if i!=0:
                    for j in range(N-i-1):
                        if i <length and j+i+1<length:
                            if order[i] in little_neighbor_dict[order[j+i+1]]:
                                edges.append(1)
                            else:
                                range_list=[order[j+i+1]]
                                dk=0
                                for d in range(N):
                                    if order[i] in range_list:
                                        dk=1./(d+1)
                                        break
                                    for x in range_list:
                                        range_list=list(set(range_list+list(set(order).intersection(set(little_neighbor_dict[x])))))
                                edges.append(dk)
                        else:
                            edges.append(0)
                elif i==0:
                    for j in range(N-i-2):
                        if i <length and j+i+2<length:
                            if order[i] in little_neighbor_dict[order[j+i+2]]:
                                edges.append(1)
                            else:
                                range_list=[order[j+i+2]]
                                dk=0
                                for d in range(N):
                                    if order[i] in range_list:
                                        dk=1./(d+1)
                                        break
                                    for x in range_list:
                                        range_list=list(set(range_list+list(set(order).intersection(set(little_neighbor_dict[x])))))
                                edges.append(dk)
                        else:
                            edges.append(0)
        
        return edges

    def sample_generate(self,m=300,N=10):
        #generate sample matrix to be trained
        #生成
        random.seed(123)
        
        #各生成M个正反例

        chosen=random.sample(self.nodes,m)
        pos=[]
        neg=[]
        for x in chosen:
            print 'pos:'
            print len(pos)
            print 'neg:'
            print len(neg)
            for y in chosen:
                if x!=y:
                    if self.edgeMatrix[x][y]==1:
                        pos.append(self.WL_algorithm((x,y),N=N))
                    else:
                        if len(pos)>=len(neg):
                            neg.append(self.WL_algorithm((x,y),N=N))
        
        pos=np.array(pos)
        neg=np.array(neg)
        np.save('./network/pos_DATA.npy',pos)
        np.save('./network/neg_DATA.npy',neg)







'''
main1=pj()
main1.load('email-Eu-core.txt',ComFileName='email-Eu-core-department-labels.txt')

main1.NeighborDict_generate()
main1.degDist()

main1.centralNodes(50)

main1.getEdgeMatrix()

#main1.visualize(range(0,100),'groundtruth')

#可视化要指定范围，超过200的节点数进行可视化的话会很慢，第一个参数输入一个可迭代对象作为可视化的节点子集



main1.balanced_kmeans()

#main1.visualize(range(100,1005),'balanced')

#main1.walkTrap()
main1.labelPropagation(thresh=10)
print main1.categories['Propagation']
#main1.visualize(range(0,100),'Random Walk')
#main1.visualize(range(0,100),'Propagation')
#main1.sample_generate(m=500,N=10)
'''
