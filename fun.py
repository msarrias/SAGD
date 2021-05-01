import numpy as np
import scipy.linalg as la
from numpy.linalg import eigh
from collections import Counter
from matplotlib import pyplot as plt
import math, random, time, pickle
import networkx as nx

def avg_simulations_sgd_results(simulations_sgd_dic):
    avg_dic = {}
    std_dic = {}
    for key, value in simulations_sgd_dic.items():
        avg_dic[key] = [np.mean(value[i]) for i in value.keys()]
        std_dic[key] = [np.std(value[i]) for i in value.keys()]
    return avg_dic, std_dic

def plot_sagd(ref_par_list, par_list, avg_dic, std_dic, save_fig_list, par = 'p' ):
    fig = plt.figure(figsize=(20, 4))
    for i in range(len(ref_par_list)):
        par0 = ref_par_list[i]
        plt.subplot(1, 5, i+1)
        plt.tight_layout()
        if par == 'p':
            plt.title(r'$p_0$ = ' + str(par0))
            plt.xlabel('p')
        else:
            plt.title(r'$k_0$ = ' + str(par0))
            plt.xlabel('k')
        plt.ylabel('SAGD')
        plt.plot(par_list, avg_dic[par0], '-o', color = 'red')
        plt.plot(par_list[np.argmin(avg_dic[par0])], 
                 np.min(avg_dic[par0]), 'x', color = 'blue')
        plt.axvline(x = par0, color = 'blue', linestyle = '--')
        minus_std = [avg_dic[par0][i] - std_dic[par0][i] 
                     for i in range(len(par_list))]
        plus_std = [avg_dic[par0][i] + std_dic[par0][i] 
                    for i in range(len(par_list))]
        plt.fill_between(par_list, minus_std, plus_std,
                         color = 'lightgrey')
    if save_fig_list[1] != 'none':
        plt.savefig(save_fig_list[0])