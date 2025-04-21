import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import math as m

# trace_dir = r'./Traces/TTST.txt'
# CAP=30

# trace_dir = r'./Traces/KVT.txt'
# CAP=198

# trace_dir = r'./Traces/EST.txt'
# CAP=64

# ARR_H = 8
np.set_printoptions(threshold=np.inf)

def read_trace(dir):

    initialized = False
    with open(dir, 'r') as f:
        raw_lines = f.readlines()

    for line in raw_lines:
        if ',' in line:
            splitted = line.split(',')
        else:
            splitted = line.split(' ')
        topop = list()
        for i in range(len(splitted)):
            item = splitted[i]

            if item == '' or item == '\n':
                topop.append(i)
            item = item.replace('[', '')
            item = item.replace(']', '')
            item = item.replace('\n', '')
            
            splitted[i] = item
        
        # pop out in reverse order
        for i in reversed(topop):
            splitted.pop(i)
        

        if len(splitted) < 5:
            continue
        else:
            if not initialized:
                # print(f'overall splitted: \n{splitted}')
                splitted = np.array(splitted).reshape((1, -1)).astype(np.int32)
                Q_paid_attention = splitted
                initialized = True
            else:
                splitted = np.array(splitted).reshape((1, -1)).astype(np.int32)
                Q_paid_attention = np.concatenate((Q_paid_attention, splitted), axis=0)

    # Q_paid_attention: dim0: N, number of tokens; dim1: _N: indexes of note worthy Ks
    return Q_paid_attention

def Ks_Qaccess(qk_index, CAP):
    # Generate the Qs(weight) that each K is attending to
    # print(f'[Ks_Qaccess] qk_index shape: {qk_index.shape}, CAP: {CAP}')
    KaccessQ = dict()
    for i in range(CAP):
        KaccessQ[i] = list()

    for i_q in range(qk_index.shape[0]):
        for i_k in range(qk_index.shape[1]):
            _k = qk_index[i_q][i_k]
            _q = i_q
            if _q in KaccessQ[_k]:
                raise ValueError(f'ERR:\t Q-{_q} already exists in K-{_k} access list')
            else:
                KaccessQ[_k].append(_q)     

    return KaccessQ    

def KQ_onehot(KaccessQ, CAP, numQ):
    # Generate the One hot Matrix indicate Q-K access (Vertical: Q_id, Horizontal: K_id)

    numQ = numQ
    numK = CAP
    onehot = np.zeros( (numQ, numK) )
    for key, value in KaccessQ.items():
        for q in value:
            onehot[q][key] = 1
    return onehot

def KQ_temporal_plot(KaccessQ):
    temporal_trace = list()

    for k, v in KaccessQ.items():
        temporal_trace.extend(v)
    
    cycles = len(temporal_trace)

    if os.path.exists(f'./figs/KQ_temporal.png'):
        os.remove(f'./figs/KQ_temporal.png')
    fig, ax = plt.subplots()
    ax.scatter(range(cycles), temporal_trace, s=0.5, marker = 'o')
    ax.set_xlabel('time step')
    ax.set_ylabel('Accessed Q id')
    fig.savefig(f'./figs/KQ_temporal.png')
    plt.close(fig)

        

def onehot_plot(onehot, name='onehot_default', xname='K id', yname='Q id'):
    # if os.path.exists(f'./figs/{name}.png'):
    #     os.remove(f'./figs/{name}.png')

    # Create scatter plots. X/Y in onehot are the coordinates to use in the scatter plot; 1 indicates a point, 0 indicates none
    
    # print(f'\n\nplotting\n')
    numrow, numcol = onehot.shape
    fig, ax = plt.subplots(figsize = (12, 8))  # Corrected order: fig, ax
    for i_row in range(onehot.shape[0]):
        # print(f'row to plot: {onehot[i_row]}')
        X = onehot[i_row]
        X = np.where(X == 1)[0]
        Y = np.ones(X.shape[0]) * i_row
        # print(f'X-axis: \n{X}')
        # print(f'Y-axis: \n{Y}')

        ax.scatter(X, Y, s=0.5, marker='o', c = 'b')
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)

    ax.set_xlim(-1, numcol+0.5)
    ax.set_ylim(-1, numrow+0.5)
    fig.savefig(f'./figs/{name}.png')
    plt.close(fig)


def dist_both1(a, b, zone_size=-1):
    # find the numbers of elements which both a and b are 1

    _both1 = a * b
    one1one0 = a + b
    one1one0 = [a % 2 for a in one1one0]
    one1one0 = np.sum(one1one0)

    both1 = 4*np.sum(_both1) - one1one0

    return both1

def cum_dist(dummy, Kin):
    # Dummy: Summed vec of sorted K, 
    agreed = np.sum(dummy * Kin)
    # dummy_bin = [a % 2 for a in dummy]
    # _newone = dummy_bin + Kin
    # _newone = [a % 2 for a in _newone]
    # newone = np.sum(_newone)
    # dist = agreed-newone
    dist = agreed
    
    return dist


def sim_sort(KQ_onehot, CAP):

    dummy_idx = 0
    dummy = KQ_onehot[:, dummy_idx]

    sortK_order = list()
    sortK_order.append(dummy_idx)

    backing = 1
    while len(sortK_order) < KQ_onehot.shape[1]:
                
        max_dist = 0
        max_id = -1
        for j in range(KQ_onehot.shape[1]):
            if j in sortK_order:
                continue


            _dist = dist_both1(KQ_onehot[:,j], dummy)
            if _dist > max_dist and _dist != 0:
                max_dist = _dist
                max_id = j
        
        if max_dist == 0:
            # no more alike rows, need to change dummy

            # print(f'[WARNING] MAX DIST = 0, changing dummy. sortedK: {len(sortK_order)}; CAP: {KQ_onehot.shape}')
            # id_dum = sortK_order[-backing]                      ## this dummy changing logic is only available
            # dummy = KQ_onehot[:][id_dum]
            # backing += 1

            for i in range(CAP):
                if i not in sortK_order:
                    id_dum = i
                    dummy = KQ_onehot[:, id_dum]
                    sortK_order.append(id_dum)
                    break

        else:
            sortK_order.append(max_id)
            backing = 1
            
            # dummy_id = sortK_order[-1]
            # dummy = KQ_onehot[:, dummy_id]

    
    return sortK_order

def cumulate_sort(KQ_onehot, numK):
    # sort based on sorted Ks. Use sorted Ks' onehot's total sum as dummy to find the most similar Ks
    numQ = KQ_onehot.shape[0]
    dummy = np.zeros(numQ)      # Dummy vec for comparison
    
    _init_id = np.random.randint(0, numK)   # Randomly select the first K

    _dummy = KQ_onehot[:, _init_id]
    dummy += _dummy

    sortK_order = list()
    sortK_order.append(_init_id)

    while len(sortK_order) < numK:
        max_dist = 0
        max_id = -1

        for j in range(KQ_onehot.shape[1]):
            if j in sortK_order:        # sorted. Skip
                continue

            _dist = cum_dist(dummy=dummy, Kin=KQ_onehot[:, j])
            if _dist > max_dist:
                max_dist = _dist
                max_id = j
        
        if max_dist == 0:
            # print(f'[WARNING] MAX DIST = 0, changing dummy. sortedK: {len(sortK_order)}; CAP: {KQ_onehot.shape}')
            for i in range(numK):
                if i not in sortK_order:
                    max_id = i
                    sortK_order.append(max_id)
                    break
        else:
            sortK_order.append(max_id)
            dummy += KQ_onehot[:, max_id]
    return sortK_order

def classify_weight(KQ_onehot, numK, heavy_size = -1):
    # classify the category of weights(K) based on the sorted Q access
    # into 1. global 2. Head heavy 3. Tail heavy

    if heavy_size == -1:
        region_length = numK // 2
    else:
        region_length = heavy_size
    globalized = list()
    head_heavy = list()
    tail_heavy = list()
    centered = list()

    # print(f'[CLASSIFY] region length: {region_length}')
    # print(f'[CLASSIFY] onehot shape: {KQ_onehot.shape}')
    for i in range(KQ_onehot.shape[0]):
        _access = KQ_onehot[i]
        head_info = np.sum(_access[:region_length])
        tail_info = np.sum(_access[-region_length:])

        if head_info == 0 and tail_info == 0:
            centered.append(i)
        elif head_info == 0:
            tail_heavy.append(i)
        elif tail_info == 0:
            head_heavy.append(i)
        else:
            globalized.append(i)

    if len(tail_heavy) > len(head_heavy):
        tail_heavy.extend(centered)
    else:
        head_heavy.extend(centered)

    return globalized, head_heavy, tail_heavy


def descend_sort(KQ_onehot):
    # perform sorting based on the number of K RDs
    K_access = KQ_onehot.sum(axis=0)
    descend_idx = np.argsort(K_access)[::-1]
    
    return descend_idx
    
def onehot_temporal_plot(onehot, name='onehot_temp_default'):
    temporal_trace = list()

    for i_col in range(onehot.shape[1]):
        for i_row in range(onehot.shape[0]):
            if onehot[i_row][i_col] == 1:
                temporal_trace.append(i_row)
    cycles = len(temporal_trace)

    if os.path.exists(f'./figs/KQ_temporal.png'):
        os.remove(f'./figs/KQ_temporal.png')
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.scatter(range(cycles), temporal_trace, s=0.5, marker = 'o')
    ax.set_xlabel('time step')
    ax.set_ylabel('Accessed Q id')
    fig.savefig(f'./figs/'+name+'.png')
    plt.close(fig)


def write_trace(KaccessQ, sortK_order, txt_dir='./Traces/KQ_sorted_trace.txt'):
    temp_trace = list()
    for k in sortK_order:
        temp_trace.extend(KaccessQ[k])
    
    if txt_dir is not None:
        with open(txt_dir, 'w') as f:
            f.write(str(temp_trace)[1:-1])
    return temp_trace   

def get_first_last_one_idx(onehot):
    # return the first and last index of 1 for a 1D array
    one_indices = np.where(onehot == 1)[0]

    first_id = one_indices[0]
    last_id = one_indices[-1]
    return first_id, last_id


def head_sort_fix(trace_dir, CAP, toplot, heavy_size = -1):

    if type(trace_dir) == str:
        # qk_index = read_trace(trace_dir)
        raise TypeError(f'[ERR] (head_sort_fix) trace_dir is of type: {type(trace_dir)}. As it should be np.arr')
    elif isinstance(trace_dir, np.ndarray):
        try:
            assert trace_dir.shape[0] == CAP, f'[ERR] (head_sort_fix) input QK shape {trace_dir.shape}, CAP={CAP}'
        except:
            print(f'trace: (shape:{trace_dir.shape})\n{trace_dir}')
            exit()
        qk_index = trace_dir 

    # pre-process: arrange in dictionary and get binary matrix
    KQ_dict = Ks_Qaccess(qk_index, CAP = CAP)       # Each K's accesses to Qs (Qs K need to be computed against)
    KQ_mat_raw = KQ_onehot(KQ_dict, CAP=CAP, numQ = qk_index.shape[0])

    cum_id = cumulate_sort(KQ_mat_raw, numK=CAP)
    sort_id = cum_id

    if toplot:
        # onehot_plot(KQ_onehot_mat, name='KQ_origin')
        print('[INFO] to_plot activated')
        onehot_plot(KQ_mat_raw[:, sort_id], name='KQ_sortQ')
    
    KQ_mat_sortK = KQ_mat_raw[:, sort_id]
    global_id, head_id, tail_id = classify_weight(KQ_mat_sortK, numK=CAP, heavy_size = heavy_size)

    condition = None

    if len(global_id) >= CAP / 2:
        condition = 'GLOBAL'
    else:
        if len(head_id) > len(tail_id):
            condition = 'HEAD'
        else:
            condition = 'TAIL'

    return KQ_mat_sortK, sort_id, global_id, head_id, tail_id, condition

def head_sort_adapt(trace_dir, CAP, div, toplot=False):
    # sort each head (QK) with adaptive Q range
    # if the result condition after sorted is 'GLOBAL', reduce the Q range to half and re-sort


    if type(trace_dir) == str:
        qk_index = read_trace(trace_dir)
    elif isinstance(trace_dir, np.ndarray):
        qk_index = trace_dir 

    numQ, numK = qk_index.shape
    KQ_dict = Ks_Qaccess(qk_index, CAP=CAP)
    KQ_mat_raw = KQ_onehot(KQ_dict, CAP = CAP, numQ = numQ)
    # print(f'KQ_MAT_RAW shape : {KQ_mat_raw.shape}')

    cum_id = cumulate_sort(KQ_mat_raw, numK = KQ_mat_raw.shape[1], CAP=CAP)
    sort_id = cum_id
    print(f'[SORT] sort_id len={len(sort_id)}')

    if toplot:
        onehot_plot(KQ_mat_raw, name='KQ_origin')
        onehot_plot(KQ_mat_raw[:, sort_id], name='KQ_sortQ')
    
    KQ_mat_sortK = KQ_mat_raw[:, sort_id]
    global_id, head_id, tail_id = classify_weight(KQ_mat_sortK, numK=numK, div=div)
    
    # if len(global_id) >= numQ * (1- 1/div):
    if len(global_id) >= numQ / 2:
        condition = 'GLOBAL'
    else:
        if len(head_id) > len(tail_id):
            condition = 'HEAD'
        else:
            condition = 'TAIL'
    
    return KQ_mat_sortK, sort_id, global_id, head_id, tail_id, condition

def subhead_sort(qkbin_fold, CAP, toplot=False, heavy_size = -1):
    assert isinstance(qkbin_fold, np.ndarray), f'[ERR] (subhead_sort) input type: {type(qkbin_fold)}'

    numQ, numK = qkbin_fold.shape

    cum_id = cumulate_sort(qkbin_fold, numK=numK) #, CAP=CAP
    sort_id = cum_id

    if toplot:
        onehot_plot(qkbin_fold, name='KQ_origin')
        onehot_plot(qkbin_fold[:, sort_id], name='KQ_sortQ')
    
    qkbin_sortK = qkbin_fold[:, sort_id]
    global_id, head_id, tail_id = classify_weight(qkbin_sortK, numK=numK, heavy_size = heavy_size)

    if len(global_id) >= numQ / 2:
        condition = 'GLOBAL'
    else:
        if len(head_id) > len(tail_id):
            condition = 'HEAD'
        else:
            condition = 'TAIL'
    return qkbin_sortK, sort_id, global_id, head_id, tail_id, condition

if __name__ == '__main__':

    qk_index = read_trace(trace_dir)

    KaccessQ = Ks_Qaccess(qk_index)
    print(f'K access Qs:')
    for k, v in KaccessQ.items():
        print(f'K {k}: {len(v)}')
        # print(f'K {k}- v:{v}')

    KQ_onehot_mat = KQ_onehot(KaccessQ)
    onehot_plot(KQ_onehot_mat, name='KQ_origin')

    descend_id = descend_sort(KQ_onehot_mat)
    KQ_onehot_dsdK = KQ_onehot_mat[:, descend_id]
    onehot_plot(KQ_onehot_dsdK, name='KQ_dsdK')

    exit()