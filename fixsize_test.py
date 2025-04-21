from KQ_trace_proc import *
import numpy as np
import math as m
import glob
import copy
import argparse

np.set_printoptions(suppress = True)

class head_info:
    def __init__(self, KQ_mat, sort_id, global_id, head_id, tail_id, condition, Korder, qk_id, div_head, div_tail, spareQ = [], spareK = []):
        self.KQ_mat = KQ_mat
        self.sort_id = sort_id
        self.global_id = global_id
        self.head_id = head_id
        self.tail_id = tail_id
        self.condition = condition
        self.Korder = Korder
        self.qk_id = qk_id

        self.div_head = div_head
        self.div_tail = div_tail

        self.spareQ = spareQ
        self.spareK = spareK

    def check_spare(self, op_type, op_val):
        assert op_type in ['Q', 'K'], f'[ERROR] (head_info) op_type must be either Q or K. Given {op_type}'
        if op_type == 'Q':
            if self.spareQ is None:
                return op_val
            # check if the input operand contain spareQ. If yes, pop spare Q(s)
            for qr in reversed(op_val):
                if qr in self.spareQ:
                    op_val.remove(qr)
            return op_val
        elif op_type == 'K':
            if self.spareK is None:
                return op_val
            # check if the input operand contain spareK. If yes, pop spare K(s)
            for kr in reversed(op_val):
                if kr in self.spareK:
                    op_val.remove(kr)
            return op_val

    def __str__(self):
        retval = f'[HEAD_{self.qk_id}] in condition {self.condition} div_head={self.div_head}; div_tail = {self.div_tail}\n'
        # retval += f'\thead_id (#={len(self.head_id)}): {self.head_id}\n'
        # retval += f'\ttail_id (#={len(self.tail_id)}): {self.tail_id}\n'
        # retval += f'\tglobal_id (#={len(self.global_id)}): {self.global_id}\n'
        retval += f'\thead_id (#={len(self.head_id)})\n'
        retval += f'\ttail_id (#={len(self.tail_id)}) \n'
        retval += f'\tglobal_id (#={len(self.global_id)}) \n'
        retval += f'\t#spareK= {len(self.spareK)}\n'
        retval += f'\t#spareQ= {len(self.spareQ)}\n'
        # retval += f'\tKorder: {self.Korder}\n'
        return retval
    
    def metadata_format(self):
        retval = f'{len(self.head_id)}, {len(self.tail_id)}, {len(self.global_id)}, {len(self.spareK)}, {len(self.spareQ)}, {self.div_head}\n'
        return retval

class INST:
    def __init__(self, OP, head_id, operand_type, operand_val):
        # operation: 'RD', 'WR', 'QINFO'(str)
        # operand_type: 'K', 'Q'        (str)
        # head_id: 0, 1, 2, ...         (int)   maybe considered as offset in addr  
        self.OP = OP
        self.operand_type = operand_type
        self.operand_val = operand_val
        self.head_id = head_id

    def __str__(self):
        # retval = f"[{self.OP}] head_{self.head_id}'s {self.operand_type}. (num={len(self.operand_val)}) Addr: {self.operand_val}"
        if self.OP == 'QINFO':
            retval = f"[{self.OP}] {self.operand_val} active Qs (head_{self.head_id})"
        else:
            retval = f"[{self.OP}] {len(self.operand_val)}-{self.operand_type}. (head_{self.head_id})"
        return retval

def folder_test(trace_dir):
    trace_files = sorted(glob.glob(trace_dir + '*'))

    filewise_QK = list()

    for _file in trace_files:
        rawQKs = list()
        # print(f'[INFO-OS] Heading into {_file}...')
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
        
        filewise_QK.append(rawQKs)

    return filewise_QK  

def QK_schedule(QKs, div, CAP, iter_cap ,heavy_size = -1, toplot = False, verbose = False):

    state = 'idle'
    num_QK = len(QKs)
    head_infos = list()
    num_resort = 0

    if heavy_size == -1:
        div_head_default = CAP//div
        div_tail_default = CAP - div_head_default
    else:
        div_head_default = heavy_size
        div_tail_default = CAP - heavy_size

    # iter_cap = 3
    
    inst_stream = dict()                        # key: timestep, value: list of INST
    global_leftover = list()                    # the list of QK that is classified as 'GLOBALized'. Process those at the end
    time_step = 0

    old_head = None                          # key: head_id, value: last head's condition
    new_head = None

    i_nexthead = 0
    last_is_global = None
    while True:
        if state is 'idle':

            # TIMESTEP 0. Initialize by writing first head's heavy and global
            escaped = False
            while True:
                qk_raw = QKs[i_nexthead]
                KQ_mat, sort_id, global_id, head_id, tail_id, condition = head_sort_fix(qk_raw, CAP=CAP, toplot = toplot, div = div, heavy_size = heavy_size)
                if verbose:
                    print(f'  [head_{i_nexthead}] num glob_id = {len(global_id)}')
                i_nexthead += 1       # debugging, not sure
                if condition is not 'GLOBAL':
                    break
                else:
                    for i in range(iter_cap):
                        print(f'[INFO] re-sorting to escape globalized QK...') if verbose else None
                        num_resort += 1
                        KQ_mat, sort_id, global_id, head_id, tail_id, condition = head_sort_fix(qk_raw, CAP=CAP, toplot = toplot, div = div, heavy_size = heavy_size-i)
                        if verbose:
                            print(f'  [head_{i_nexthead-1}] num glob_id = {len(global_id)}')

                        if condition is not 'GLOBAL':
                            escaped = True
                            div_head_escape = heavy_size - i
                            div_tail_escape = CAP - div_head_escape
                            break
                    if escaped:
                        print(f'[INFO] escape SUCCESSFULLY')  if verbose else None
                        break
                    else:
                        _head = head_info(KQ_mat, sort_id, global_id, head_id, tail_id, condition, sort_id, qk_id = i_nexthead-1, div_head = div_head_default, div_tail = div_tail_default)
                        global_leftover.append(_head)
                        print(f'[WARNING] escape FAILED. globalized QK found at head_{i_nexthead-1}') if verbose else None
                        print(_head)

                        if i_nexthead == num_QK:
                            # All Qks deemed as globalized.
                            return inst_stream, global_leftover
                # i_nexthead += 1

            if condition == 'TAIL':
                Korder = sort_id[::-1]
            else:
                # HEAD/GLOBAL/BALANCED cases use the sorted id directly
                Korder = sort_id
            
            _div_head = div_head_escape if escaped else div_head_default
            _div_tail = div_tail_escape if escaped else div_tail_default
                
            new_head = head_info(KQ_mat, sort_id, global_id, head_id, tail_id, condition, Korder, qk_id = i_nexthead-1, div_head = _div_head, div_tail = _div_tail)
            head_infos.append(new_head)
            # print(new_head)

            inst_stream[time_step] = list()
            # WR heavy
            if condition in ['HEAD', 'BALANCED']:
                _inst = INST(OP='WR', head_id = i_nexthead-1, operand_type = 'Q', operand_val = new_head.head_id + new_head.global_id)
                inst_stream[time_step].append(_inst)
            elif condition is 'TAIL':
                _inst = INST(OP='WR', head_id = i_nexthead-1, operand_type = 'Q', operand_val = new_head.tail_id + new_head.global_id)
                inst_stream[time_step].append(_inst)
            state = 'intohead'


        elif state is 'intohead':
            # start RDing into the head: RD first 1/3. & WR the leftover Qs of the head
            inst_stream[time_step] = list()
            if new_head.condition in ['HEAD', 'BALANCED']:
                # last QK is head-heavy. WR last QK's tail, RD first 2/3 of last QK's K

                # WR head's leftover Q
                _inst = INST(OP='WR', head_id = new_head.qk_id, operand_type = 'Q', operand_val = new_head.tail_id)
                inst_stream[time_step].append(_inst)

            elif new_head.condition is 'TAIL':
                # last QK is tail-heavy. WR last QK's head, RD first 2/3 of last QK's (reversed) K

                # WR oldhead's leftover Q
                _inst = INST(OP='WR', head_id = new_head.qk_id, operand_type = 'Q', operand_val = new_head.head_id)
                inst_stream[time_step].append(_inst)

            # RD the 1st 1/3 of K (0-33%)
            _inst = INST(OP='RD', head_id = new_head.qk_id, operand_type = 'K', operand_val = new_head.Korder[:new_head.div_head])
            inst_stream[time_step].append(_inst)

            state = 'midsthead'

        elif state is 'midsthead':
            #  RDing the middle part of K (e.g. 1/3 ~ 2/3)
            # this part of K is Assumed necessary for both TAIL-heaby, HEAD-heavy and GLOBAL Qs and is the optimization target (as small as possible)
            inst_stream[time_step] = list()
            _inst = INST(OP='RD', head_id = new_head.qk_id, operand_type='K', operand_val = new_head.Korder[new_head.div_head:new_head.div_tail])
            inst_stream[time_step].append(_inst)

            state = 'outtahead'

        elif state is 'outtahead':
            # RDing the last 1/3 K (66% ~ 100%) of the head & 
            # Start writing Heavy + Glob part of the new head

            # holder update
            old_head = new_head             
            if i_nexthead == num_QK:
                inst_stream[time_step] = list()

                _inst = INST(OP='RD', head_id = old_head.qk_id, operand_type='K', operand_val = old_head.Korder[old_head.div_tail:])
                inst_stream[time_step].append(_inst)
                break
            else:
                # old_head = new_head

                escaped = False
                while True:
                    qk_raw = QKs[i_nexthead]
                    KQ_mat, sort_id, global_id, head_id, tail_id, condition = head_sort_fix(qk_raw, CAP=CAP, toplot = toplot, div = div, heavy_size = heavy_size)
                    if verbose:
                        print(f'  [head_{i_nexthead}] num glob_id = {len(global_id)}')

                    i_nexthead += 1       # not sure
                    if condition is not 'GLOBAL':
                        last_is_global = False
                        break
                    else:
                        for i in range(iter_cap):
                            print(f'[INFO] re-sorting to escape globalized QK... ({i} unit away from default heavy_size)') if verbose else None
                            num_resort += 1
                            KQ_mat, sort_id, global_id, head_id, tail_id, condition = head_sort_fix(qk_raw, CAP=CAP, toplot = toplot, div = div, heavy_size = heavy_size - i)
                            if verbose:
                                print(f'  [head_{i_nexthead-1}] num glob_id = {len(global_id)}')

                            if condition is not 'GLOBAL':
                                escaped = True
                                div_head_escape = heavy_size - i
                                div_tail_escape = CAP - div_head_escape
                                break
                        if escaped:
                            print(f'[INFO] escape SUCCESSFULLY') if verbose else None
                            last_is_global = False
                            break
                        else:
                            _head = head_info(KQ_mat, sort_id, global_id, head_id, tail_id, condition, sort_id, qk_id = i_nexthead-1, div_head = div_head_default, div_tail = div_tail_default)
                            global_leftover.append(_head)
                            print(f'[WARNING] escape FAILED. globalized QK found at head_{i_nexthead-1}') if verbose else None
                            print(_head)

                            if i_nexthead == num_QK:
                                last_is_global = True
                                break
                    # i_nexthead += 1

                if last_is_global:
                    # reached the last QK and it is globalized. Wrap up with RDing leftover Ks 
                    state = 'wrapup'
                    new_head = old_head
                    old_head = None
                    continue

                if condition == 'TAIL':
                    Korder = sort_id[::-1]
                else:
                    # HEAD/GLOBAL/BALANCED cases use the sorted id directly
                    Korder = sort_id

                _div_head = div_head_escape if escaped else div_head_default
                _div_tail = div_tail_escape if escaped else div_tail_default

                new_head = head_info(KQ_mat, sort_id, global_id, head_id, tail_id, condition, Korder, qk_id = i_nexthead-1, div_head = _div_head, div_tail = _div_tail)
                # print(new_head)
                head_infos.append(new_head)

                # RD oldhead's K
                inst_stream[time_step] = list()

                _inst = INST(OP='RD', head_id = old_head.qk_id, operand_type='K', operand_val = old_head.Korder[old_head.div_tail:])
                inst_stream[time_step].append(_inst)

                # WR new_head's heavy Q & global Q
                if new_head.condition in ['HEAD', 'BALANCED']:
                    _operand_val = new_head.head_id + new_head.global_id
                elif new_head.condition is 'TAIL':
                    _operand_val = new_head.tail_id + new_head.global_id

                _inst = INST(OP='WR', head_id = new_head.qk_id, operand_type='Q', operand_val = _operand_val)
                inst_stream[time_step].append(_inst)
                
                state = 'intohead'

        elif state is 'wrapup':
            # wrapup state occur when in outtahead state, the left Qs are all 'GLOBAL'.
            # RD the leftover Ks of the last (not 'GLOBAL') head
            
            if last_is_global:
                _inst = INST(OP='RD', head_id = new_head.qk_id, operand_type='K', operand_val = new_head.Korder[new_head.div_tail:])
                inst_stream[time_step].append(_inst)
                break

        time_step += 1
    return inst_stream, global_leftover, head_infos, num_resort

def global_wrapup(global_leftover, inst_stream):
    # execute head that is deemed as globalized
    # naive implementation 

    _time = max(inst_stream.keys()) + 1 if len(inst_stream) != 0 else 0

    for _head in global_leftover:
        inst_stream[_time] = list()
        _inst = INST(OP='WR', head_id = _head.qk_id, operand_type='Q', operand_val = _head.head_id + _head.global_id + _head.tail_id)
        inst_stream[_time].append(_inst)
        _time += 1

        inst_stream[_time] = list()
        _inst = INST(OP='RD', head_id = _head.qk_id, operand_type='K', operand_val = _head.Korder)
        inst_stream[_time].append(_inst)
        _time += 1
    # return inst_stream


def inst_stream_process(inst_stream):
    for k, v in inst_stream.items():
        if len(v) > 2:
            idx = list()
            for i in range(len(v)):
                if v[i].OP == 'WR':
                    idx.append(i)

            _val = v[idx[1]].operand_val 
            for q in _val:
                v[idx[0]].operand_val.append( q )
            v.pop(idx[1])
    return inst_stream
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--heavy_size', type=int, default=-1, help='heavy size during sorting')
    args = parser.parse_args()


    available_unit = 30
    heavy_size = -1
    verbose = False
    head_print = False
    output_trace_dir = r'./OutTrace/'
    output_hd_dir = r'./OutHead/'

    if not os.path.exists(output_trace_dir):
        os.makedirs(output_trace_dir)
    if not os.path.exists(output_hd_dir):
        os.makedirs(output_hd_dir)

    # ----------- All Head QK Trace files ---------------#
    trace_dir = r'./Traces/TTST_all/'
    CAP=30
    div=3
    _heavy_size = 15
    heavy_size = args.heavy_size if args.heavy_size != -1 else _heavy_size
    iter_cap = heavy_size
    output_trace_dir += 'TTST.txt'
    output_hd_dir += 'TTSThd.txt'

    # a folder dir means to test all head in all file
    filewise_QKs = folder_test(trace_dir)
    total_resort = 0

    filewise_accel = list()
    f = open(output_trace_dir, 'w')
    f2 = open(output_hd_dir, 'w')
    f2.write(f'#head_id, #tail_id, #glob_id, #spareQ, #spareK, Heavy Size, INIT_Size {heavy_size} \n')


    for i in range(len(filewise_QKs)):
        if verbose:
            print(f'---- File Index {i} ---- ')

        f.write(f'---- File Index {i} ---- \n')

        QKs = filewise_QKs[i]
        inst_stream0, glob_leftover, head_infos, num_resort = QK_schedule(QKs, div, CAP, iter_cap = iter_cap, heavy_size = heavy_size, toplot = False, verbose = verbose)
        total_resort += num_resort

        print(f'[INFO] #GLOB = {len(glob_leftover)}')
        if len(glob_leftover) != 0:   
            print(f'[INFO] #GLOB = {len(glob_leftover)} out of {len(QKs)} QKs')
            global_wrapup(global_leftover=glob_leftover, inst_stream = inst_stream0)
        inst_stream = inst_stream_process(copy.deepcopy(inst_stream0))

        head_infos += glob_leftover
        for _head in head_infos:
            f2.write(f'{_head.metadata_format()}')

        if head_print:
            for _head in head_infos:
                print(_head)
                # f.write(str(_head) + '\n')

        if verbose:
            for k, v in inst_stream.items():
                print(f'Timestep {k}:')
                for inst in v:
                    print(f'\t{inst}')

        for k, v in inst_stream.items():
            f.write(f'Timestep {k}:\n')
            for inst in v:
                f.write(f'\t{inst}\n')
            
    f.close()
    print(f'---- SUMMARY ----')
    print(f'Total Resort times: {total_resort} (heavy_size = {heavy_size})')
    print(f'---- END ----')