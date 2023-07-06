import numpy as np
import math

def bonferonni_holm_pval_correction(P):
    P_ordered = np.squeeze(P).argsort()
    P_adj = []
    P_adj_ord = np.zeros_like(P)
    for count, idx in enumerate(P_ordered):
        p = P[idx]
        p_adj = p*(len(P) - (count+1) + 1)
        if count > 0:
            p_adj = max(p_adj, P_adj[count-1])
        p_adj = min(p_adj, 1)
        P_adj.append(p_adj)
        P_adj_ord[idx] = p_adj
    return np.array(P_adj_ord)

def fdr_pval_correction(P):
    # below method argsort si incorrect?
    P = np.asarray(P)
    # P_ordered = np.squeeze(P).argsort()
    # fdr = P * len(P) / P_ordered
    # fdr[fdr > 1] = 1
    from scipy.stats import rankdata
    ranked_p_values = rankdata(P)
    fdr = P * len(P) / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr

def bonferonni_pval_correction(P):
    P_ordered = np.squeeze(P).argsort()
    P_adj = []
    P_adj_ord = np.zeros_like(P)
    for count, idx in enumerate(P_ordered):
        p = P[idx]
        p_adj = p*len(P)
        p_adj = min(p_adj, 1)
        P_adj.append(p_adj)
        P_adj_ord[idx] = p_adj
    return np.array(P_adj_ord)

def convert_p_val(P):
    if math.isnan(P):
        p = '****'
    elif P <= 0.00001:
        p = '****'
    elif P <= 0.0001:
        p = '***'
    elif P <= 0.001:
        p = '**'
    elif P <= 0.01:
        p = '**'
    elif P <= 0.05:
        p = '*'
    else:
        p = 'ns'
    return p   

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
