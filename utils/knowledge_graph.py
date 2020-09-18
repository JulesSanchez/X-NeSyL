import numpy as np 
from .config import *
import torch, sys
import networkx as nx 

sys.path.append("../MonuMAI-AutomaticStyleClassification")
from monumai.monument import Monument

styles = FOLDERS_DATA
archi_features = [el for sublist in list(Monument.ELEMENT_DIC.values()) for el in sublist]

knowledge_graph = np.ones( (len(archi_features),len(styles)) ) * -1

for i in range(len(Monument.ELEMENT_DIC)):
    local_el = archi_features[i]
    for k in range(len(list(Monument.TRUE_ELEMENT_DIC.keys()))):
        style = list(Monument.TRUE_ELEMENT_DIC.keys())[k]
        if local_el in Monument.TRUE_ELEMENT_DIC[style]:
            knowledge_graph[i,k] = 1

def compare_shap_and_KG(shap_values, true_labels, threshold = 0):
    contrib = np.zeros((len(true_labels),len(archi_features)))
    for k in range(len(true_labels)):
        local_kg = knowledge_graph[:,true_labels[k]]
        contrib[k] = shap_values[true_labels[k]][k]*local_kg
    contrib[contrib>-threshold] = 0
    return contrib

def get_bbox_weight(shap_values,is_exponential=False,h=1):
    shap_array = np.dstack((shap_values[0],shap_values[1],shap_values[2],shap_values[3]))
    contrib = np.ones((shap_array.shape[0],shap_array.shape[1]+1,shap_array.shape[2]))
    for i in range(len(shap_array)):
        FG = nx.Graph()
        contrib[i,1:,:] = shap_array[i,:,:]*knowledge_graph
    contrib[contrib>=0] = 1
    if not is_exponential:
        contrib[contrib<0] = -h*contrib[contrib<0] + 1
    else:
        contrib[contrib<0] = np.exp(-h*contrib[contrib<0])
    contrib = torch.from_numpy(contrib)
    contrib.requires_grad = False
    return contrib.type(torch.cuda.FloatTensor)

def reduce_shap(contrib,is_exponential = False,h=1):
    shap_coeff = np.zeros(len(contrib))
    if not is_exponential:
        shap_coeff = -h*np.min(contrib,axis=1) + 1
    else :
        shap_coeff = np.exp(-h*np.min(contrib,axis=1))
    shap_coeff = torch.from_numpy(shap_coeff)
    shap_coeff.requires_grad = False
    return shap_coeff.type(torch.cuda.FloatTensor)


def filter_KG(knowledge_graph, facade_graph):
    nodes = facade_graph.nodes
    return knowledge_graph.subgraph(nodes)

def compare_node(n1,n2):
    return n1['name'] == n2['name']
        

def distance(filtered_graph, facade_graph):
    d = 0
    for edge in filtered_graph.edges:
            if not (facade_graph.has_edge(edge[0],edge[1]) or facade_graph.has_edge(edge[1],edge[0])):
                d += 1
    for edge in facade_graph.edges:
            if not (filtered_graph.has_edge(edge[0],edge[1]) or filtered_graph.has_edge(edge[1],edge[0])):
                d += 1
    return d

names = [el for sublist in list(Monument.ELEMENT_DIC.values()) for el in sublist]
element_dic = Monument.TRUE_ELEMENT_DIC
styles = Monument.STYLES_HOTONE_ENCODE

def make_KG(for_causal=False):
    
    index_dic = {}
    reversed_index_dic = {}
    index = 0
    
    KG = nx.Graph()
    
    for s in range(len(styles)):
        KG.add_node(index, name=styles[s])
        index_dic[index] = styles[s]
        reversed_index_dic[styles[s]] = index
        index += 1
    
    if for_causal:
        for i in range(len(styles)):
            for j in range(i+1,len(styles)):
                KG.add_edge(reversed_index_dic[styles[i]],reversed_index_dic[styles[j]])
                
                
    keys_el_dic =  list(element_dic.keys())         
    for i in range(len(keys_el_dic)):
        for el in element_dic[keys_el_dic[i]]:
            if el not in reversed_index_dic:
                KG.add_node(index, name=el)
                KG.add_edge(index,reversed_index_dic[styles[i]])
                index_dic[index] = el
                reversed_index_dic[el] = index
                index += 1
            else:
                local_index = reversed_index_dic[el]
                KG.add_edge(local_index,reversed_index_dic[styles[i]])
    
    index_dic[index+1] = 'contrib'
    reversed_index_dic['contrib'] = index+1
    
    return KG, index_dic, reversed_index_dic
    
def GED_metric(shap_values,threshold=0.01):
    KG, index_dic, reversed_index_dic = make_KG(False)
    d_tot = 0
    shap_array = np.dstack((shap_values[0],shap_values[1],shap_values[2],shap_values[3]))
    for i in range(len(shap_array)):
        FG = nx.Graph()
        for k in range(shap_array.shape[-1]):
            facade = np.copy(shap_array[i,:,k])
            facade[facade<threshold] = 0
            facade[facade>=threshold] = 1
            facade = facade.astype(np.uint8)
            style_index = reversed_index_dic[styles[k]]
            if np.sum(facade) > 0:
                FG.add_node(style_index, name=styles[k])
                for j in range(len(facade)):
                    if facade[j]:
                        index = reversed_index_dic[names[j]]
                        FG.add_node(index, name=names[j])
                        FG.add_edge(style_index, index)
        sub_KG = filter_KG(KG,FG)
        d = distance(sub_KG, FG)
        d_tot += d
    return float(d_tot)/len(shap_array)



