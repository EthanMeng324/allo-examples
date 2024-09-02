
from allo.ir.types import int4, int8, int16, int32, index, Int, UInt
import allo
import numpy as np

def conditional_dataflow[
    TyA: Int,
    TyB: Int,
    TyC: Int,
    N: int32,
](
    inst: "TyA",
    fifo_1: "TyB[N]",
    fifo_2: "TyC[N]",
):
    buffer: int32[N]
    
    if inst == 0:
        for i in range(N - 1, name="stage0"):
            buffer[i + 1] += buffer[i]
    elif inst == 1:
        for i in range(N, name="stage1"):
            fifo_1[i] = buffer[i]
    else:
        for i in range(N, name="stage2"):
            fifo_2[i] = buffer[i]   
            
def conditional_compute[
    TyA: Int,
    N: int32,
](
    inst: "TyA",
    input: "int8[N]",
    output: "int8[N]",
):
    for i in range(N, name="compute"):
        tmp: int8 = input[i]
        tmp_out: int8
        if inst == 0:
            tmp_out = tmp * 2
        else:
            input0: int4 = tmp[0: 4]
            tmp_out = input0 * 2
        output[i] = tmp_out
        
def conditional_memory_layout[
    TyA: Int,
    TyB: Int,
    TyC: Int,
    N: int32,
](
    inst: "TyA[2]",
    input: "TyB[N, N]",
    output: "TyC[N, N]",
):
    stage_a: TyA = inst[0]
    stage_b: TyA = inst[1]
    buffer: int32[N, N]
    
    for i, j in allo.grid(N, N):
        if stage_a == 0:
            buffer[i, j] = input[i, j]
        else:
            buffer[j, i] = input[i, j]
    
    for i, j in allo.grid(N, N):
        if stage_b == 0:
            output[i, j] = buffer[i, j]
        else:
            output[i, j] = buffer[j, i]

# s = allo.customize(conditional_dataflow, instantiate=[int32, int16, int16, 5])
# mod = s.build(target="llvm")
# inst = 1
# fifo_1 = np.ones(5).astype(np.int16)
# fifo_2 = np.ones(5).astype(np.int16)
# mod(inst, fifo_1, fifo_2)
# print(fifo_1)
# print(fifo_2)

# s = allo.customize(conditional_compute, instantiate=[int32, 5])
# mod = s.build(target="llvm")
# inst = 1
# input = np.array([18, 2, 3, 4, 5]).astype(np.int8)
# output = np.zeros(5).astype(np.int8)
# mod(inst, input, output)
# print(output)

s = allo.customize(conditional_memory_layout, instantiate=[int32, int16, int16, 3])
mod = s.build(target="llvm")
inst = np.array([0, 1])
input = np.arange(1, 10).reshape((3, 3)).astype(np.int16)
output = np.zeros(9).reshape((3, 3)).astype(np.int16)
mod(inst, input, output)
print(output)

