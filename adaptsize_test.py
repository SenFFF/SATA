from KQ_trace_proc import *
import numpy as np
import math as m
import glob
import copy
from fixsize_test import head_info, INST, calc_activeQ

np.set_printoptions(suppress = True)


def folder_test(trace_dir):
    trace_files = sorted(glob.glob(trace_dir + '*'))
    rawQKs = list()

    for _file in trace_files:
        print(f'[INFO-OS] Heading into {_file}...')
        with open(_file, 'r') as f:
            raw_lines = f.readlines()

        holder = list()
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


            if len(splitted) == 0:
                #  one head is all processed. 
                if len(holder) == 0:
                    continue
                else:
                    holder = np.array(holder).astype(np.int32)
                    rawQKs.append(holder)
                    holder = list()         # re-init
            else:
                holder.append(splitted)
    return rawQKs, trace_files

def simple_hw_estimate(inst_stream):

    cycle_per_timestep = list()

    for k, v in inst_stream.items():
        # timestep latency
        if len(v) == 1:
            cycle_per_timestep.append(len(v[0].operand_val))
        else:
            _ops = set()
            for _inst in v:
                _ops.add(_inst.OP)

            _lat1 = len(v[0].operand_val)
            _lat2 = len(v[1].operand_val)

            if len(_ops) == 1:
                cycle_per_timestep.append(_lat1 + _lat2)
            else:
                cycle_per_timestep.append(max(_lat1, _lat2))

    return cycle_per_timestep

def verify_correctness(inst_stream, QK, i_foldK):
    # verify the correctness of scheduling within a single head.
    Qsize_sub = 36
    offset_subQ = Qsize_sub
    Ksize_sub = 18
    offset_subK = Ksize_sub

    K_lowcap = i_foldK * Ksize_sub
    K_upcap = (i_foldK+1) * Ksize_sub

    QK_monitor = copy.deepcopy(QK)

    # clearup the demands outside this K fold.
    for i_q in range(QK.shape[0]):
        for i_k in range(QK.shape[1]):
            _elem = QK_monitor[i_q, i_k]
            if _elem < K_lowcap or _elem >= K_upcap:
                QK_monitor[i_q, i_k] = -1

    activeQ = list()
    # for sub_stream in inst_stream:
    #     # sub_stream: the inst_stream of a group of subheads (one dummy column of binary QK_mat)

    for time, insts in inst_stream.items():

        for _inst in insts:
            i_sub = _inst.head_id
            Q_offset = i_sub * offset_subQ


            # mark newly written Qs as 'active'
            if _inst.OP == 'WR':
                _Qs = _inst.operand_val
                Qs = [Q_offset + q for q in _Qs]
                activeQ.extend(Qs)

            elif _inst.OP == 'RD':
                _Ks = _inst.operand_val
                Ks = [k+K_lowcap for k in _Ks]

                for _k in Ks:

                    for _q in activeQ:

                        # if _k in QK_monitor[_q]:
                        if np.any(QK_monitor[_q] == _k):
                            k_id = np.where(QK_monitor[_q] == _k)[0][0]
                            QK_monitor[_q, k_id] = -1
                        else:
                            # raise ValueError(f'[ERROR] K_{_k} not found in Q_{_q}\'s demand list: \n{QK_monitor[_q]}')
                            pass
    
    if np.all(QK_monitor == -1):
        print(f'[INFO] All demands are satisfied')
    else:
        print(f'[INFO] ERROR Some demands are not satisfied')
        print(QK_monitor)
        exit()
                            


def subregion_division(QK, fold_stepQ, fold_stepK, id_foldK, CAP, div, iter_cap, heavy_size = -1, verbose = False):
    # The QK here refer to QK of a single head. Cause now each QK is too large to be sorted at one go; instead they are treated in sub-regions
    # QK: shape= [All Q, needed K] (e.g. 198, 90)
    num_Qfold = m.ceil(QK.shape[0] / fold_stepQ)
    num_Kfold = m.ceil(QK.shape[1] / fold_stepK)                       #  = number of subheads

    if heavy_size == -1:
        div_delta = CAP//div
        div_head_default = div_delta
        div_tail_default = CAP - div_delta
    else:
        div_head_default = heavy_size
        div_tail_default = fold_stepK - heavy_size

    subheads = list()                       # key: subhead id; value: subhead's head_info
    glob_subheads = list()

    qk_dict = Ks_Qaccess(QK, CAP=CAP)
    qk_bin = KQ_onehot(qk_dict, CAP=CAP, numQ = QK.shape[0])
    # this func only deals with one 'dummy column' in the overall large binary QK_mat
    startK = id_foldK * fold_stepK
    endK = (id_foldK+1) * fold_stepK
    fold_offset = id_foldK * fold_stepK
    # fold_offset = 0
    for i_fold in range(num_Qfold):

        fold_CAP = int(CAP/num_Kfold)
        startQ = i_fold * fold_stepQ
        endQ = (i_fold+1) * fold_stepQ

        qk_bin_fold = qk_bin[startQ:endQ, startK: endK]       
        # returned QK_mat is the sorted binary mat of the current Q/K fold 
        QK_mat, sort_id, global_id, head_id, tail_id, condition = subhead_sort(qkbin_fold = qk_bin_fold, CAP=fold_CAP, toplot=False, div=div, heavy_size=heavy_size)

        # Try to repeat sorting to avoid GLOBAL case
        escaped = False
        heavy_offset = -1
        if condition is not 'GLOBAL':
            if verbose:
                print(f'[INFO] fold_{i_fold}: {condition}. sortid len={len(sort_id)} (#glob={len(global_id)}; #head={len(head_id)}; #tail={len(tail_id)})')

        else:
            # Global. Try resort to avoid GLOBAL
            for i_iter in range(iter_cap):
                if verbose:
                    if i_iter == 0:
                        print(f'[INFO] re-sorting to escape global QK... ({i_iter} unit away from default heavy_size)', end=' ')
                    else:
                        print(f'{i_iter}', end=' ')
                        
                QK_mat, sort_id, global_id, head_id, tail_id, condition = subhead_sort(qkbin_fold = qk_bin_fold, CAP=fold_CAP, toplot=False, div=div, heavy_size=heavy_size - i_iter)
                if condition is not 'GLOBAL':
                    escaped = True
                    heavy_offset = i_iter
                    if verbose:
                        print(f'\n[INFO] fold_{i_fold} escape GLOBAL SUCCESS: {condition} (#glob={len(global_id)}; #head={len(head_id)}; #tail={len(tail_id)})', end=' ')
                    break   
            
        if condition is 'GLOBAL':
            # if after iter_cap, still GLOBAL, escape failed.
            if verbose:
                print(f'\n[INFO] fold_{i_fold} escape GLOBAL FAILED: {condition} (#glob={len(global_id)}; #head={len(head_id)}; #tail={len(tail_id)})', end=' ')

        ## Zero Elimination
        Ksum = QK_mat.sum(axis=0)
        zeroK = np.where(Ksum == 0)[0]
        Qsum = QK_mat.sum(axis=1)
        zeroQ = np.where(Qsum == 0)[0]
        # print(f'\t\t(REDUNDANT [Q, K] = {len(zeroQ), len(zeroK)}) spareQ: {list(zeroQ)}, spareK: {list(zeroK)} ')
        
        _div_head = div_head_default if not escaped else heavy_size - heavy_offset
        _div_tail = div_tail_default if not escaped else fold_stepK - heavy_size + heavy_offset

        Korder = sort_id[::-1] if condition is 'TAIL' else sort_id

        _spareQ = list(zeroQ) if len(zeroQ) > 0 else None
        _spareK = list(zeroK) if len(zeroK) > 0 else None
        _head = head_info(QK_mat, sort_id, global_id, head_id, tail_id, condition, Korder, qk_id = i_fold + fold_offset, div_head = _div_head, div_tail = _div_tail,
                          spareQ = _spareQ, spareK = _spareK)

        if condition is 'GLOBAL':
            glob_subheads.append(_head)   
        else:
            subheads.append(_head)
        
    subheads.extend(glob_subheads)
    return subheads    

def subhead_schedule(subheads):

    # print(f'[INFO] (default) div_head = {div_head_default}; div_tail = {div_tail_default}')

    old_sub = None
    new_sub = None
    state = 'idle'

    num_subhead = len(subheads)
    time_step = 0
    subhead_insts = dict()

    i_sub = 0
    while True:

        if state is 'idle':
            # first subhead, write heavy & global
            _sub = subheads[i_sub]
            i_sub += 1

            new_sub = _sub
            subhead_insts[time_step] = list()

            # WRing weights(Q) that will can retire early
            if new_sub.condition in ['HEAD']:   # ['HEAD', 'BALANCED']
                _q_toWR = new_sub.head_id + new_sub.global_id
                _q_toWR = new_sub.check_spare(op_type = 'Q', op_val = _q_toWR)  # filter the Q that is not accessed by any K in this fold
            elif new_sub.condition is 'TAIL':
                _q_toWR = new_sub.tail_id + new_sub.global_id
                _q_toWR = new_sub.check_spare(op_type = 'Q', op_val = _q_toWR)
            else:
                raise ValueError(f'Unknown condition {new_sub.condition} found')

            _inst = INST(OP='WR', head_id = new_sub.qk_id, operand_type='Q', operand_val = _q_toWR)
            subhead_insts[time_step].append(_inst)
            
            state = 'intohead'
        
        elif state is 'intohead':
            # Already written the heavy+glob part of subhead's Q. Write the rest of Q &
            # RD the first 1/3 (or 0 ~ heavy_size) of K
            subhead_insts[time_step] = list()
            
            if new_sub.condition in ['HEAD']:  #['HEAD', 'BALANCED']
                _q_toWR = new_sub.tail_id
                _q_toWR = new_sub.check_spare(op_type = 'Q', op_val = _q_toWR)
            elif new_sub.condition is 'TAIL':
                _q_toWR = new_sub.head_id
                _q_toWR = new_sub.check_spare(op_type = 'Q', op_val = _q_toWR)
            else:
                raise ValueError(f'Unknown condition {new_sub.condition} found')
            
            _k_toRD = new_sub.Korder[:new_sub.div_head]
            _k_toRD = new_sub.check_spare(op_type = 'K', op_val = _k_toRD)

            _inst = INST(OP='WR', head_id = new_sub.qk_id, operand_type='Q', operand_val = _q_toWR)
            subhead_insts[time_step].append(_inst)
            _inst = INST(OP='RD', head_id = new_sub.qk_id, operand_type='K', operand_val = _k_toRD)
            subhead_insts[time_step].append(_inst)

            state = 'midsthead'

            # if i_sub == num_subhead:
            #     if (new_sub.condition is 'HEAD' and len(new_sub.head_id) == CAP) or new_sub.condition is 'TAIL' and len(new_sub.tail_id) == CAP:
            #         break
            #     state = 'wrapup'
            
        elif state is 'midsthead':
            # read the middle [div_head:div_tail] of Ks
            subhead_insts[time_step] = list()

            _k_toRD = new_sub.Korder[new_sub.div_head:new_sub.div_tail]
            _k_toRD = new_sub.check_spare(op_type = 'K', op_val = _k_toRD)

            _inst = INST(OP='RD', head_id = new_sub.qk_id, operand_type='K', operand_val = _k_toRD)
            subhead_insts[time_step].append(_inst)

            state = 'outtahead'

        elif state is 'outtahead':
            # Finish RDing the light part of old_sub's K, and start WRing heavy part of new_sub's Q
            old_sub = new_sub
            subhead_insts[time_step] = list()

            try:
                _sub = subheads[i_sub]
                i_sub += 1
            except:
                print(f'[INFO] i_sub = {i_sub} ; num_subhead = {num_subhead}')
                exit()

            # if i_sub == num_subhead:
            #     state = 'wrapup'
            # else:
            #     i_sub += 1

            new_sub = _sub

            # RD old_sub's K
            _k_toRD = old_sub.Korder[old_sub.div_tail:] 
            _k_toRD = old_sub.check_spare(op_type = 'K', op_val = _k_toRD)

            _inst = INST(OP='RD', head_id = old_sub.qk_id, operand_type='K', operand_val = _k_toRD)
            subhead_insts[time_step].append(_inst)

            # if state is 'wrapup':
            #     break

            # WR new_sub's heavy Q & global Q
            if new_sub.condition in ['HEAD']: #['HEAD', 'BALANCED']
                _q_toWR = new_sub.head_id + new_sub.global_id
            elif new_sub.condition is 'TAIL':
                _q_toWR = new_sub.tail_id + new_sub.global_id
            elif new_sub.condition is 'GLOBAL':
                # next sub_head in GLOBAL condition. No early retire and scheduling possible. 
                state = 'glob_wrapup'
                i_sub -= 1
                continue
            else:
                raise ValueError(f'Unknown condition {new_sub.condition} found')
            _q_toWR = new_sub.check_spare(op_type = 'Q', op_val = _q_toWR)

            _inst = INST(OP='WR', head_id = new_sub.qk_id, operand_type='Q', operand_val = _q_toWR)
            subhead_insts[time_step].append(_inst)

            if i_sub == num_subhead:
                state = 'wrapup'
            else:
                state = 'intohead'

        elif state is 'glob_wrapup':
            # the rest sub-heads should all be in 'GLOBAL' condition. Schedule direct RD->WR workflow till end
            # print(f'\n *** HEADING INTO GLOB_WRAPUP STAGE ***\n')
            num_global = num_subhead - i_sub 
            # print(f'\t\ti_sub = {i_sub} ; num_subhead = {num_subhead}')

            for _ in range(num_global):
                subhead_insts[time_step] = list()
                _q_toWR = new_sub.head_id + new_sub.global_id + new_sub.tail_id
                _q_toWR = new_sub.check_spare(op_type = 'Q', op_val = _q_toWR)
                _inst = INST(OP='WR', head_id = new_sub.qk_id, operand_type='Q', operand_val = _q_toWR)
                subhead_insts[time_step].append(_inst)
                time_step += 1

                subhead_insts[time_step] = list()
                _k_toRD = new_sub.Korder
                _k_toRD = new_sub.check_spare(op_type = 'K', op_val = _k_toRD)
                _inst = INST(OP='RD', head_id = new_sub.qk_id, operand_type='K', operand_val = _k_toRD)
                subhead_insts[time_step].append(_inst)
                time_step += 1

                if i_sub < num_subhead-1:
                    i_sub += 1
                    new_sub = subheads[i_sub]
                    # assert new_sub.condition is 'GLOBAL', f'[ERROR] subhead condition is not GLOBAL but end up in glob_wrapup state'
            
            assert i_sub == num_subhead-1, f'[ERROR] i_sub={i_sub} != len(_subheads)={num_subhead}'
            
            # Initialize for next head's subheads schedule
            state = 'idle'
            break
        
        elif state is 'wrapup':
            # NO global subhead. Normal wrapup
            old_sub = new_sub

            subhead_insts[time_step] = list()
            _k_toRD = old_sub.Korder[:old_sub.div_head]
            _k_toRD = old_sub.check_spare(op_type = 'K', op_val = _k_toRD)

            _inst = INST(OP='RD', head_id = old_sub.qk_id, operand_type = 'K', operand_val=_k_toRD)
            subhead_insts[time_step].append(_inst)

            if old_sub.condition in ['HEAD', 'BALANCED']:
                _q_toWR = old_sub.tail_id
            elif old_sub.condition is 'TAIL':
                _q_toWR = old_sub.head_id
            _q_toWR = old_sub.check_spare(op_type = 'Q', op_val = _q_toWR)

            _inst = INST(OP='WR', head_id = old_sub.qk_id, operand_type = 'Q', operand_val = _q_toWR)
            subhead_insts[time_step].append(_inst)

            # Finish the last subhead's K
            time_step += 1  
            subhead_insts[time_step] = list()
            _k_toRD = old_sub.Korder[old_sub.div_head:old_sub.div_tail]
            _k_toRD = old_sub.check_spare(op_type = 'K', op_val = _k_toRD)
            _inst = INST(OP='RD', head_id = old_sub.qk_id, operand_type = 'K', operand_val=_k_toRD)
            subhead_insts[time_step].append(_inst)

            time_step += 1
            subhead_insts[time_step] = list()
            _k_toRD = old_sub.Korder[old_sub.div_tail:]
            _k_toRD = old_sub.check_spare(op_type = 'K', op_val = _k_toRD)
            _inst = INST(OP='RD', head_id = old_sub.qk_id, operand_type = 'K', operand_val = _k_toRD)
            subhead_insts[time_step].append(_inst)
            state = 'idle'
            break

        time_step += 1

    return subhead_insts
            
def subhead_display(subhead_insts, latency):

    for time, insts in subhead_insts.items():
        print(f'time_{time}: (lat={latency[time]})')
        for _inst in insts:
            print(f'\t{_inst}')

def subhead_zeroexam(QK_mat_sub):
    # detect Qs that is not accessed by any Ks within the sub-fold
    # Assume the input is already the subfold's QK_mat (e.g. 198 * 9)
    numQ, numK = QK_mat_sub.shape

    num_idleQ = 0
    for i in range(numQ):
        _QK = QK_mat_sub[i, :]
        if np.all(_QK == 0):
            # print(f'[info] Q_{i} has no access')
            num_idleQ += 1
    print(f'[INFO] #idleQ = {num_idleQ} / {numQ}')


def simple_hw_estimate(inst_stream, QK_subhead, CAP, div):
    # print(f'[INFO] qk_subhead type: {type(QK_subhead)}; element type : {type(QK_subhead[0])}')
    if type(QK_subhead[0]) is list:
        print(f'QK_subhead[0]: \n{QK_subhead[0]}')

    cycle_per_timestep = list()
    # onethird_RD_TU = round(CAP / div)
    
    subid_offset = QK_subhead[0].qk_id
    for k, v in inst_stream.items():
        # timestep latency
        extra_TU = 0

        for _inst in v:
            head_ids = [i.head_id for i in v]
            if len(set(head_ids)) > 1:
                # RDing old subhead's K & WRing new subhead's Q, possibility exist that stall cycles will be introduced
                _sub_rd = v[0].head_id
                _sub_wr = v[1].head_id
                assert v[0].OP is 'RD' and v[1].OP is 'WR', f'[ERROR] RD-WR pair order mistaken'

                num_retire = -1
                if QK_subhead[_sub_rd-subid_offset].condition in ['HEAD', 'BALANCED']:
                    num_retire = len(QK_subhead[_sub_rd-subid_offset].head_id)
                else:
                    num_retire = len(QK_subhead[_sub_rd-subid_offset].tail_id)
                
                RDing_time = len(v[0].operand_val)

                _extra_TU = RDing_time - num_retire
                extra_TU = _extra_TU if _extra_TU > 0 else 0
                
        if len(v) == 1:
            cycle_per_timestep.append(len(v[0].operand_val))
        else:
            _ops = set()
            for _inst in v:
                _ops.add(_inst.OP)

            _lat1 = len(v[0].operand_val)
            try:
                _lat2 = len(v[1].operand_val) if v[1].OP is not 'QINFO' else 0
            except:
                print(f'[ERROR] v[1] = {v[1]}')

            if len(_ops) == 1:
                cycle_per_timestep.append(_lat1 + _lat2 + extra_TU)
            else:
                cycle_per_timestep.append(max(_lat1, _lat2) + extra_TU)

    return cycle_per_timestep

def single_QK_test(QK, div, CAP, fold_stepQ, fold_stepK, heavy_size, iter_cap, f, verbose = False, equalize = False):
    # f: opened file handle

    # QK_binary = KQ_onehot(Ks_Qaccess(QK, CAP=CAP), CAP=CAP, numQ = QK.shape[0])
    numK_effective = QK.shape[0]
    num_foldK = m.ceil(numK_effective / fold_stepK)

    QK_subheads_subwise = list()

    inst_stream_all = dict()

    i_subhd = 0
    for i_foldK in range(num_foldK):

        QK_subhead = subregion_division(QK, fold_stepQ = fold_stepQ, fold_stepK = fold_stepK, id_foldK = i_foldK,
                                        CAP=CAP, div=div, iter_cap = iter_cap, heavy_size=heavy_size, verbose = verbose)

        # for _sub in QK_subhead: 
        #     print(f'[INFO] subhead_{_sub.qk_id}: {len(_sub.head_id)}-{len(_sub.tail_id)}-{len(_sub.global_id)}')

        _inst_stream0 = subhead_schedule(QK_subhead)

        _inst_stream = calc_activeQ(copy.deepcopy(_inst_stream0), QK_subhead)

        # f.write(f'---- fold Index {i_subhd} ----\n')
        for k, v in _inst_stream.items():
            f.write(f'Timestep {k}:\n')
            for inst in v:
                f.write(f'\t{inst}\n')
        f.write('\n')
        i_subhd += 1

        lat = simple_hw_estimate(_inst_stream0, QK_subhead, CAP = fold_stepK, div=div)

        if verbose:
            subhead_display(_inst_stream, lat)
        
            print(f'[INFO] K_fold_ID_{i_foldK}:')
            print(f'[HW ESTI]:\t\t{sum(lat)} TU \n')

        QK_subheads_subwise.extend(QK_subhead)
        inst_stream_all.update(_inst_stream)

    all_lat = simple_hw_estimate(inst_stream_all, QK_subheads_subwise, CAP = fold_stepK, div=div)

    lat_sum = sum(all_lat)
    overall_accelerate = (CAP*2 / lat_sum - 1) * 100

    print(f'[INFO] overall latency (max): {lat_sum}')
    print(f'[INFO] accelerate% = {overall_accelerate:.2f}%')

    print('\n')

    return all_lat, overall_accelerate, None, None

if __name__ == '__main__':

    verbose = False
    output_trace_dir = r'./OutTrace/'

    if not os.path.exists(output_trace_dir):
        os.makedirs(output_trace_dir)

    # ----------- Single Head QK Trace ---------------#
    # trace_dir = r'./Traces/TTST.txt'
    # CAP=30

    # trace_dir = r'./Traces/KVT.txt'
    # CAP=198

    # trace_dir = r'./Traces/EST.txt'
    # CAP=64

    # ----------- All Head QK Trace files ---------------#
    trace_dir = r'./Traces/KVT_all/'
    CAP=198
    div=3
    fold_stepQ = 10
    ## fold_stepK = 12
    fold_stepK = fold_stepQ
    heavy_size = 5
    ## iter_cap = 6
    iter_cap = heavy_size
    output_trace_dir += 'KVT.txt'

    # trace_dir = r'./Traces/DRS_all/'
    # CAP=48
    # div=3
    # fold_stepQ = 8
    # # fold_stepK = 12
    # fold_stepK = fold_stepQ
    # heavy_size = 4
    # iter_cap = heavy_size
    # output_trace_dir += 'DRS.txt'

    # trace_dir = r'./Traces/TTST_all/'
    # CAP=30
    # div=3
    # fold_stepQ = 10
    # # fold_stepK = 30
    # fold_stepK = fold_stepQ
    # heavy_size = 5
    # iter_cap = heavy_size

    if 'txt' in trace_dir:
        # means that only a single head trace is given
        KQ_mat, sort_id, global_id, head_id, tail_id, condition = head_sort_fix(trace_dir=trace_dir, CAP=CAP, toplot = False, div=div)
        # print(f'KQ mat shape:{KQ_mat.shape}')
    else:
        # ---- All Global infos ---- #
        # a folder dir means to test all head in all file

        QKs, _ = folder_test(trace_dir)
        print(f'[INFO] #QKs = {len(QKs)}')
        print(f'[INFO] sample QK shape: {QKs[0].shape[0]}x{QKs[0].shape[1]}')

        latency = list()
        accelerate = list()
        latency_equalize = list()
        accelerate_equalize = list()

        # onehot_plot(KQ_onehot(Ks_Qaccess(QKs[0], CAP=CAP), CAP=CAP, numQ = QKs[0].shape[0]), name='QK_0')
        headwise_accel = []
        f = open(output_trace_dir, 'w')

        equalize = False
        for i_qk in range(len(QKs)):
            print(f'[INFO] QK _ {i_qk}')
            _QK = QKs[0]
            # QK_binary = KQ_onehot(Ks_Qaccess(QK, CAP=CAP), CAP=CAP, numQ = QK.shape[0])
            numK_effective = _QK.shape[0]

            f.write(f'---- qk Index {i_qk} ----\n')
            lat, accel, lat_eql, accel_eql = single_QK_test(_QK, div, CAP, fold_stepQ, fold_stepK, heavy_size, iter_cap, f, verbose = verbose, equalize = equalize)

            latency.append(lat)
            accelerate.append(accel)
            if equalize:
                latency_equalize.append(lat_eql)
                accelerate_equalize.append(accel_eql)

        print(f'------- SUMMARY -------')
        print(f'\tQK_id\t|\tRaw accelerate %\t|\t Equalized accelerate %')
        for i in range(len(QKs)):
            if equalize:
                print(f'\t{i}\t|\t{accelerate[i]:.2f}\t|\t{accelerate_equalize[i]:.2f}')
            else:
                print(f'\t{i}\t|\t{accelerate[i]:.2f}\t|\t-')

        print(f'[AVERAGE]:')
        print(f'\t Raw acceleration % = {sum(accelerate) / len(accelerate):.2f}% (max={max(accelerate):.2f}%; min={min(accelerate):.2f}%)')
        if equalize:
            print(f'\t Equalized acceleration % = {sum(accelerate_equalize) / len(accelerate_equalize):.2f}% (max={max(accelerate_equalize):.2f}%; min={min(accelerate_equalize):.2f}%)')
        f.close()
        