import numpy as np
import scipy.linalg as la
from numpy.linalg import eigh
from collections import Counter
from matplotlib import pyplot as plt
import math, random, time
import networkx as nx

class graphs_distance:
    def __init__(self, ref, W_ref, W_dic, norm_type = "scale_and_shift", verbose = True):
        self.ref = ref
        self.W_ref = W_ref
        self.W_dic = W_dic
        self.norm_type = norm_type
        self.verbose = verbose
        self.N_ref, _  = self.W_ref.shape
        self.params_list = list(self.W_dic.keys())
        self.N_replicates = len(W_dic[self.params_list[0]])
        self.eps = 1e-10
        if self.norm_type not in ["scale_and_shift", "norm_wrt_volume", "norm_wrt_avg_ctd"]:
            raise Exception('choose a valid normalization form')
    
    def reset_W_ref(self, new_ref, new_W_ref):
        self.W_ref = new_W_ref
        self.ref = new_ref
        self.compute_CTDs_ref()
        self.compute_asymp_CTDs_ref()
    
    def reset_normalization(self, new_normalization):
        if new_normalization not in ["scale_and_shift", "norm_wrt_volume", "norm_wrt_avg_ctd"]:
            raise Exception('choose a valid normalization form')
        else:
            self.norm_type = new_normalization
        
    def compute_D(self, matrix):
        return np.diag(sum(matrix))
    
    def compute_C_matrix(self, eigv, eigvc, sqrtVol):
        N = len(eigv)
        V_T = np.vstack(eigvc[1:])
        sqrt_inv_W = np.diag(1 / np.sqrt(np.asarray(eigv[1:]) + 1e-10))
        CTE = np.dot(sqrtVol * sqrt_inv_W, V_T).T
        C = np.zeros((N,N))
        for xi in range(N):
            for xj in range(xi, N):
                C[xi, xj] = np.sqrt(sum((CTE[xi] -  CTE[xj])**2)) 
                C[xj, xi] = C[xi, xj]
        return C
    
    def compute_asymp_C_matrix(self, D_matrix):
        N, _ = D_matrix.shape
        C = np.zeros((N, N))
        for xi in range(N):
            for xj in range(xi, N):
                C[xi, xj] = np.sum(D_matrix) * ((1/ D_matrix[xi, xi]) + (1/ D_matrix[xj, xj]))
                C[xj, xi] = C[xi, xj]
        return C
    
    def solve_and_sort_std_eigenvalue_problem(self, matrix):
        eigv, eigvc = la.eig(matrix)
        sorted_eig =  sorted(zip(eigv.real, eigvc.T), key=lambda x: x[0])
        return [[elm[i] for elm in sorted_eig] for i in [0,1]]
    
    def compute_CTDs_ref(self):
        self.D_ref = self.compute_D(self.W_ref)
        self.L_ref = self.D_ref - self.W_ref
        self.Vol_ref = np.sum(self.D_ref)
        self.eigvs_ref, self.eigvec_ref = self.solve_and_sort_std_eigenvalue_problem(self.L_ref)
        self.C_ref = self.compute_C_matrix(self.eigvs_ref, self.eigvec_ref, np.sqrt(self.Vol_ref))
        if self.verbose:
            print(f'Reference model CTD matrix completed')
            
    def compute_asymp_CTDs_ref(self):
        self.D_ref = self.compute_D(self.W_ref)
        self.Vol_ref = np.sum(self.D_ref)
        self.C_asymp_ref = self.compute_asymp_C_matrix(self.D_ref)
        if self.verbose:
            print(f'Asymptotic rerence model CTD matrix completed')
            
    def compute_asymp_CTDs_dic(self):
        self.C_asymp_dic = {}
        self.Eval_dic = {}
        tic = time.time()
        for parm in self.params_list:
            C_dic_temp = {}
            for replicate in range(self.N_replicates):
                D_temp = self.compute_D(self.W_dic[parm][replicate])
                C_dic_temp[replicate] = self.compute_asymp_C_matrix(D_temp)
            self.C_asymp_dic[parm] = C_dic_temp
            print('||' , end = '')
        print('')
        if self.verbose: 
            tac = time.time()
            print(f'CTD matrices of the {self.N_replicates * len(self.params_list)}' \
                  f' models completed. Process took {(tac-tic)/60} minutes')
            print('')
                                                   
    def compute_CTDs_dic(self):
        self.C_dic = {}
        self.Eval_dic = {}
        tic = time.time()
        for parm in self.params_list:
            C_dic_temp = {}
            Eval_dic_temp = {}
            for replicate in range(self.N_replicates):
                W_temp = self.W_dic[parm][replicate]
                D_temp = self.compute_D(W_temp)
                eigv, eigvc = self.solve_and_sort_std_eigenvalue_problem(D_temp - W_temp)
                Eval_dic_temp[replicate] = eigv
                C_dic_temp[replicate] = self.compute_C_matrix(eigv, eigvc, np.sqrt(np.sum(D_temp)))
            self.C_dic[parm] = C_dic_temp
            self.Eval_dic[parm] = Eval_dic_temp
            print('||' , end = '')
        print('')
        if self.verbose: 
            tac = time.time()
            print(f'CTD matrices of the {self.N_replicates * len(self.params_list)}' \
                  f' models completed. Process took {(tac-tic)/60} minutes')
            print('')
    
    def normalize_ctds(self, list_values, Vol):
        if self.norm_type == "scale_and_shift":
            return ((list_values - min(list_values)) / (max(list_values) - min(list_values)))
        if self.norm_type == "norm_wrt_volume":
            return list_values / Vol
        if self.norm_type == "norm_wrt_avg_ctd":
            return np.asarray(list_values) / np.mean(list_values)
        
    def Kruglov_distance(self, vi, vj):
        u = sorted(list(set([*vi , *vj])))
        vi_counter = Counter(vi)
        vj_counter = Counter(vj)
        vi_count = vi_counter.get(u[0], 0)
        vj_count = vj_counter.get(u[0], 0)
        cdf_distance = 0
        Ni = len(vi)
        Nj = len(vj)
        for idx, sorted_elt in enumerate(u[1:]):
            # adding rectangle area, L * W
            abs_dif = abs((vi_count / Ni) - (vj_count / Nj))
            cdf_distance += (u[idx + 1] - u[idx]) * abs_dif
            vi_count += vi_counter.get(sorted_elt, 0)
            vj_count += vj_counter.get(sorted_elt, 0)
        return cdf_distance
    
    def compute_SAGD(self):
        self.CTDs_ref = list(self.C_ref[np.triu_indices(self.N_ref, k = 1)])
        self.NormCTDs_ref = self.normalize_ctds(self.CTDs_ref, self.Vol_ref)
        tic = time.time()
        if self.verbose: 
            print('Calulating the SAGD between:')
            print(f' G({self.ref}) and G(~), as an average  ' \
                  f'of the SAGD between G({self.ref}) and the {self.N_replicates} replicates')
        self.rep_SAGD = {}
        self.time = []
        self.CTDs_dic = {}
        for idx, key in enumerate(self.params_list):
            tic_ = time.time() 
            replicate_distance = []
            ctds_temp = {}
            for rep in range(self.N_replicates):
                C_j  = self.C_dic[key][rep]
                CTDs_j = list(C_j[np.triu_indices(C_j.shape[0], k = 1)])
                NormCTDs_j = self.normalize_ctds(CTDs_j, np.sum(self.W_dic[key][rep]))
                ctds_temp[rep] = NormCTDs_j
                replicate_distance.append(self.Kruglov_distance(self.NormCTDs_ref, NormCTDs_j))
            self.CTDs_dic[key] = ctds_temp
            self.rep_SAGD[key] = replicate_distance
            tac_ = time.time()
            time_ = (tac_ - tic_) / 60
            self.time.append(time_)
        tac = time.time()
        if self.verbose:
            print(f'Process took {(tac-tic)/60}') 
            
    def compute_asymp_SAGD(self):
        self.CTDs_asymp_ref = list(self.C_asymp_ref[np.triu_indices(self.N_ref, k = 1)])
        self.NormCTDs_asymp_ref = self.normalize_ctds(self.CTDs_asymp_ref, self.Vol_ref)
        tic = time.time()
        if self.verbose: 
            print('Calulating the asymptotic SAGD between:')
            print(f' G({self.ref}) and G(~), as an average  ' \
                  f'of the SAGD between G({self.ref}) and the {self.N_replicates} replicates')
        self.rep_asymp_SAGD = {}
        self.time = []
        self.CTDs_asymp_dic = {}
        for idx, key in enumerate(self.params_list):
            tic_ = time.time() 
            replicate_distance = []
            ctds_temp = {}
            for rep in range(self.N_replicates):
                C_j  = self.C_asymp_dic[key][rep]
                CTDs_j = list(C_j[np.triu_indices(C_j.shape[0], k = 1)])
                NormCTDs_j = self.normalize_ctds(CTDs_j, np.sum(self.W_dic[key][rep]))
                ctds_temp[rep] = NormCTDs_j
                replicate_distance.append(self.Kruglov_distance(self.NormCTDs_asymp_ref, NormCTDs_j))
            self.CTDs_asymp_dic[key] = ctds_temp
            self.rep_asymp_SAGD[key] = replicate_distance
            tac_ = time.time()
            time_ = (tac_ - tic_) / 60
            self.time.append(time_)
        tac = time.time()
        if self.verbose:
            print(f'Process took {(tac-tic)/60}')
    
    def ASD(self, eigv_i, eigv_j):
        return np.sqrt(np.sum((eigv_i - eigv_j)**2))
    
    def compute_ASD(self):
        tic = time.time()
        self.rep_ASD = {}
        self.time = []
        if self.verbose: 
            print('Calulating the ASD between:')
            print(f' G({self.ref}) and G(~), as an average  ' \
                  f'of the ASD between G({self.ref}) and the {self.N_replicates} replicates')
        for idx, key in enumerate(self.params_list):
            tic_ = time.time() 
            replicate_distance = []
            for rep in range(self.N_replicates):
                replicate_distance.append(self.ASD(np.asarray(self.eigvs_ref),
                                                   np.asarray(self.Eval_dic[key][rep])))
            self.rep_ASD[key] = replicate_distance
            tac_ = time.time()
            time_ = (tac_ - tic_) / 60
            self.time.append(time_)
        tac = time.time()
        if self.verbose:
            print(f'Process took {(tac-tic)/60} minutes') 
    
    
    