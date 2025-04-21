from KQ_trace_proc import *
import numpy as np
import math as m
import glob
import copy
from fixsize_test import head_info, INST, calc_activeQ
import argparse
import time

np.set_printoptions(suppress = True)


def folder_test(trace_dir):
    trace_files = sorted(glob.glob(trace_dir + '*'))
    rawQKs = list()

    for _file in trace_files:
        print(f'[INFO-OS] Heading into {_file}...')
        with open(_file, 'r') as f:
            raw_lines = f.readlines()
        # print(f'[INFO] {_file} has {len(raw_lines)} lines')

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

            # print(f'len splitted = {len(splitted)}')
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
    _sort_latency = 0
    num_Qfold = m.ceil(QK.shape[0] / fold_stepQ)
    num_Kfold = m.ceil(QK.shape[1] / fold_stepK)                       #  = number of subheads
    num_resort = 0

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

    for i_fold in range(num_Qfold):
        fold_CAP = int(CAP/num_Kfold)
        startQ = i_fold * fold_stepQ
        endQ = (i_fold+1) * fold_stepQ

        qk_bin_fold = qk_bin[startQ:endQ, startK: endK]       
        if qk_bin_fold.shape[0] == 0:
            raise ValueError(f'[ERROR] startQ == endQ. QK_mat is empty. i_fold = {i_fold}; startQ = {startQ}; endQ = {endQ}; startK = {startK}; endK = {endK}')

        # returned QK_mat is the sorted binary mat of the current Q/K fold 
        _start = time.time()
        QK_mat, sort_id, global_id, head_id, tail_id, condition = subhead_sort(qkbin_fold = qk_bin_fold, CAP=fold_CAP, toplot=False, div=div, heavy_size=heavy_size)
        _end = time.time()
        _sort_latency += _end - _start

        # if id_foldK == 0:
        # onehot_plot(QK_mat, name=f'QK_{i_fold}')

        # Try to repeat sorting to avoid GLOBAL case
        escaped = False
        heavy_offset = -1
        if condition is not 'GLOBAL':
            if verbose:
                print(f'[INFO] fold_{i_fold}: {condition}. sortid len={len(sort_id)} (#glob={len(global_id)}; #head={len(head_id)}; #tail={len(tail_id)})')

        else:
            # Global. Try resort to avoid GLOBAL
            for i in range(iter_cap):
                if verbose:
                    if i == 0:
                        print(f'[INFO] re-sorting to escape global QK... ({i} unit away from default heavy_size)', end=' ')
                    else:
                        print(f'{i}', end=' ')
                        
                QK_mat, sort_id, global_id, head_id, tail_id, condition = subhead_sort(qkbin_fold = qk_bin_fold, CAP=fold_CAP, toplot=False, div=div, heavy_size=heavy_size - i)
                num_resort += 1
                if condition is not 'GLOBAL':
                    escaped = True
                    heavy_offset = i
                    if verbose:
                        print(f'\n[INFO] fold_{i_fold} escape GLOBAL SUCCESS: {condition} (#glob={len(global_id)}; #head={len(head_id)}; #tail={len(tail_id)})', end=' ')
                    break   
            
        if condition is 'GLOBAL':
            # if after iter_cap, still GLOBAL, escape failed.
            if verbose:
                print(f'\n[INFO] fold_{i_fold} escape GLOBAL FAILED: {condition} (#glob={len(global_id)}; #head={len(head_id)}; #tail={len(tail_id)})', end=' ')

        Ksum = QK_mat.sum(axis=0)
        zeroK = np.where(Ksum == 0)[0]
        Qsum = QK_mat.sum(axis=1)
        zeroQ = np.where(Qsum == 0)[0]
        # print(f'\t\t(REDUNDANT [Q, K] = {len(zeroQ), len(zeroK)}) spareQ: {list(zeroQ)}, spareK: {list(zeroK)} ')
        
        _div_head = div_head_default if not escaped else heavy_size - heavy_offset
        _div_tail = div_tail_default if not escaped else fold_stepK - heavy_size + heavy_offset

        Korder = sort_id[::-1] if condition is 'TAIL' else sort_id

        _spareQ = list(zeroQ) if len(zeroQ) > 0 else []
        _spareK = list(zeroK) if len(zeroK) > 0 else []
        _head = head_info(QK_mat, sort_id, global_id, head_id, tail_id, condition, Korder, qk_id = i_fold, div_head = _div_head, div_tail = _div_tail,
                          spareQ = _spareQ, spareK = _spareK)

        if condition is 'GLOBAL':
            glob_subheads.append(_head)   
        else:
            subheads.append(_head)
        
    subheads.extend(glob_subheads)
    return subheads, num_resort, _sort_latency

def subhead_schedule(subheads, div, CAP, heavy_size = -1):


    # if heavy_size == -1:
    #     div_delta = CAP//div
    #     div_head_default = div_delta
    #     div_tail_default = CAP - div_delta
    # else:
    #     div_head_default = heavy_size
    #     div_tail_default = CAP - heavy_size

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
            if new_sub.condition in ['HEAD', 'BALANCED']:
                _q_toWR = new_sub.head_id + new_sub.global_id
                _q_toWR = new_sub.check_spare(op_type = 'Q', op_val = _q_toWR)
            elif new_sub.condition is 'TAIL':
                _q_toWR = new_sub.tail_id + new_sub.global_id
                _q_toWR = new_sub.check_spare(op_type = 'Q', op_val = _q_toWR)

            _inst = INST(OP='WR', head_id = new_sub.qk_id, operand_type='Q', operand_val = _q_toWR)
            subhead_insts[time_step].append(_inst)
            
            state = 'intohead'
        
        elif state is 'intohead':
            # Already written the heavy+glob part of subhead's Q. Write the rest of Q &
            # RD the first 1/3 (or 0 ~ heavy_size) of K
            subhead_insts[time_step] = list()
            
            if new_sub.condition in ['HEAD', 'BALANCED']:
                _q_toWR = new_sub.tail_id
                _q_toWR = new_sub.check_spare(op_type = 'Q', op_val = _q_toWR)
            elif new_sub.condition is 'TAIL':
                _q_toWR = new_sub.head_id
                _q_toWR = new_sub.check_spare(op_type = 'Q', op_val = _q_toWR)
            
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

            if state is 'wrapup':
                break

            # WR new_sub's heavy Q & global Q
            if new_sub.condition in ['HEAD', 'BALANCED']:
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

    # print(f'[INFO] ---- SCHEDULING ENDS ----')
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
    cycle_per_timestep = list()
    # onethird_RD_TU = round(CAP / div)

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
                if QK_subhead[_sub_rd].condition in ['HEAD', 'BALANCED']:
                    num_retire = len(QK_subhead[_sub_rd].head_id)
                else:
                    num_retire = len(QK_subhead[_sub_rd].tail_id)
                
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
            _lat2 = len(v[1].operand_val)

            if len(_ops) == 1:
                cycle_per_timestep.append(_lat1 + _lat2 + extra_TU)
            else:
                cycle_per_timestep.append(max(_lat1, _lat2) + extra_TU)

    return cycle_per_timestep

def single_QK_test(QK, div, CAP, fold_stepQ, fold_stepK, heavy_size, iter_cap, f, f2, verbose = False, equalize = False):
    # f: opened file handle: temporal trace
    # f2: opened file handle: head info
    sort_latency = 0

    # QK_binary = KQ_onehot(Ks_Qaccess(QK, CAP=CAP), CAP=CAP, numQ = QK.shape[0])
    numK_effective = QK.shape[0]
    num_foldK = m.ceil(numK_effective / fold_stepK)

    inst_stream_subwise = dict()
    QK_subheads_subwise = list()

    lat_Kfold = list()
    num_resort = 0
    subheads = list()

    i_subhd = 0
    for i_foldK in range(num_foldK):

        QK_subhead, _num_resort, _sort_lat = subregion_division(QK, fold_stepQ = fold_stepQ, fold_stepK = fold_stepK, id_foldK = i_foldK,
                                        CAP=CAP, div=div, iter_cap = iter_cap, heavy_size=heavy_size, verbose = verbose)
        num_resort += _num_resort
        subheads.extend(QK_subhead)
        sort_latency += _sort_lat

        # print(f'QK_subhead: ')
        # for _sub in QK_subhead:
        #     print(f'\t{_sub}')
        # exit()
        _inst_stream0 = subhead_schedule(QK_subhead, div, CAP = fold_stepK, heavy_size=heavy_size)
        _inst_stream = calc_activeQ(copy.deepcopy(_inst_stream0), QK_subhead)

        # f.write(f'---- fold Index {i_subhd} ----\n')
        for k, v in _inst_stream.items():
            f.write(f'Timestep {k}:\n')
            for inst in v:
                f.write(f'\t{inst}\n')
        f.write('\n')
        i_subhd += 1

        for _sub in QK_subhead:
            f2.write(f'{_sub.metadata_format()}')

        if verbose:
            subhead_display(_inst_stream, lat)
        
            print(f'[INFO] K_fold_ID_{i_foldK}:')
            print(f'[HW ESTI]:\t\t{sum(lat)} TU \n')

        # lat_Kfold.append(sum(lat))

        # _QK_subK = QK_binary[:, i_foldK * fold_stepK: (i_foldK+1) * fold_stepK]
        # verify_correctness(QK = QK, inst_stream = inst_stream, i_foldK = i_foldK)

        inst_stream_subwise[i_foldK] = _inst_stream
        QK_subheads_subwise.append(QK_subhead)

    # overall_lat = max(lat_Kfold)
    # overall_accelerate = (CAP*2 / overall_lat - 1) * 100

    # print(f'[INFO] K-fold wise latency: \n{lat_Kfold}')
    # print(f'[INFO] overall latency (max): {overall_lat}')
    # print(f'[INFO] best case latency (min): {min(lat_Kfold)}')
    # print(f'[INFO] accelerate% = {overall_accelerate:.2f}%')

    # _lat_wr = CAP
    # _lat_rd = m.ceil(CAP/fold_stepQ) * fold_stepK
    # _direct_delay = _lat_wr  + _lat_rd
    # columnwise_accelerate = [(_direct_delay / lat - 1) * 100 for lat in lat_Kfold]
    

    # print(f'[INFO] columnwise accelerate%: ')
    # for i in columnwise_accelerate:
    #     print(f'{i:.2f}', end=', ')
    # print('\n')

    return -1, -1, None, None, num_resort, subheads, sort_latency

if __name__ == '__main__':

    argparse = argparse.ArgumentParser()
    argparse.add_argument('--heavy_size', type = int, default = -1)
    argparse.add_argument('--fold_step', type = int, default = -1)
    args = argparse.parse_args()
    heavy_size_arg = args.heavy_size
    fold_step_arg = args.fold_step

    verbose = True
    output_trace_dir = r'./OutTrace/'
    output_head_dir = r'./OutHead/'

    if not os.path.exists(output_trace_dir):
        os.makedirs(output_trace_dir)
    if not os.path.exists(output_head_dir):
        os.makedirs(output_head_dir)


    # ----------- All Head QK Trace files ---------------#
    trace_dir = r'./Traces/KVT_Deit_Tiny/'
    CAP=198
    div=3
    fold_stepQ = 33
    fold_stepQ = fold_step_arg if fold_step_arg != -1 else fold_stepQ
    fold_stepK = fold_stepQ
    heavy_size = int(fold_stepQ / 2)
    heavy_size = heavy_size_arg if heavy_size_arg != -1 else heavy_size
    iter_cap = heavy_size
    output_trace_dir += 'KVT.txt'           # specific to KVT_Deit_Tiny
    output_head_dir += 'KVThd.txt'

    # trace_dir = r'./Traces/KVT_Deit_Base/'
    # CAP=198
    # div=3
    # fold_stepQ = 18
    # fold_stepQ = fold_step_arg if fold_step_arg != -1 else fold_stepQ
    # fold_stepK = fold_stepQ
    # heavy_size = int(fold_stepQ / 2)
    # heavy_size = heavy_size_arg if heavy_size_arg != -1 else heavy_size
    # iter_cap = heavy_size
    # output_trace_dir += 'KVTBase.txt'
    # output_head_dir += 'KVTBasehd.txt'

    # trace_dir = r'./Traces/DRS_all/'
    # CAP=48    
    # div=3
    # fold_stepQ = 6
    # fold_stepQ = fold_step_arg if fold_step_arg != -1 else fold_stepQ
    # fold_stepK = fold_stepQ
    # heavy_size = int(fold_stepQ / 2)
    # heavy_size = heavy_size_arg if heavy_size_arg != -1 else heavy_size
    # iter_cap = heavy_size
    # output_trace_dir += 'DRS.txt'
    # output_head_dir += 'DRShd.txt'

    # ---- All Global infos ---- #
    # a folder dir means to test all head in all file



    sort_lat_total = 0
    num_resort_tot = 0
    QKs, _ = folder_test(trace_dir)
    print(f'[INFO] #QKs = {len(QKs)}')
    print(f'[INFO] sample QK shape: {QKs[0].shape}; fold size: {fold_stepQ}')

    latency = list()
    accelerate = list()
    latency_equalize = list()
    accelerate_equalize = list()

    # onehot_plot(KQ_onehot(Ks_Qaccess(QKs[0], CAP=CAP), CAP=CAP, numQ = QKs[0].shape[0]), name='QK_0')
    headwise_accel = []
    f = open(output_trace_dir, 'w')
    f2 = open(output_head_dir, 'w')
    f2.write(f'#head_id, #tail_id, #glob_id, #spareQ, #spareK, Heavy Size, INIT_Size {fold_stepQ} \n')
    start_time = time.time()
    

    for i_qk in range(len(QKs)):
        # print(f'[INFO] QKs _ \n{QKs[i_qk]}')
        
        _QK = QKs[i_qk]
        # QK_binary = KQ_onehot(Ks_Qaccess(QK, CAP=CAP), CAP=CAP, numQ = QK.shape[0])
        numK_effective = _QK.shape[0]

        f.write(f'---- qk Index {i_qk} ----\n')
        lat, accel, lat_eql, accel_eql, num_resort, subheads, sort_lat = single_QK_test(_QK, div, CAP, fold_stepQ, fold_stepK, heavy_size, iter_cap, f, f2, verbose = False, equalize = False)
        num_resort_tot += num_resort
        sort_lat_total += sort_lat

        latency.append(lat)
        accelerate.append(accel)
        # print(f'[INFO] QK_{i_qk} subheads:')
        # for _sub in subheads:
        #     print(f'\t{_sub}')

    end_time = time.time()
    elapse = end_time - start_time
    print(f'[INFO] Scheduling elapse time = {elapse:.2f} sec')
    print(f'\tsort latency = {sort_lat_total:.2f} s')
    # print(f'------- SUMMARY -------')
    # print(f'\tQK_id\t|\tRaw accelerate %\t|\t Equalized accelerate %')
    # for i in range(len(QKs)):
    #     print(f'\t{i}\t|\t{accelerate[i]:.2f}\t|\t-')

    # print(f'[AVERAGE]:')
    # print(f'\t Raw acceleration % = {sum(accelerate) / len(accelerate):.2f}% (max={max(accelerate):.2f}%; min={min(accelerate):.2f}%) (fold_step = {fold_stepQ})')
    print(f'\t Num_resort = {num_resort_tot}')
    f.close()
    f2.close()
        