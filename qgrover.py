# https://stackoverflow.com/questions/67420191/how-is-data-encoded-in-an-oracle-on-a-quantum-circuit

import numpy as np
import random
from random import sample
import math

from qiskit import circuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, EstimatorV2 as Estimator
from qiskit.circuit.library.standard_gates import MCXGate


#clause = 3
def oracle_n(qc, clause, clause_qbits, n, output):
    clause_binary = format(clause, f'0{n}b')[::-1] # 0011
    qc.append(circuit.library.MCXGate(n, ctrl_state=clause_binary), clause_qbits[0:n] + [output]) 


def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    # was = qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.mcx(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    return qc

def forward_qram_n(qc, register_qubits, data, n, matrix):    
    for i in range(0, len(matrix)):
        mask = format(matrix[i], f'0{n}b')
        mask_i = format(i, f'0{n}b')
        integer_bit = 0
        for bit in mask_i:
            if bit == '0':
                qc.x(register_qubits[integer_bit])
            integer_bit += 1
        
        integer_bit = 0
        for bit in mask:
            if bit == '1':
                qc.append(circuit.library.MCXGate(n), register_qubits[0:n] + [data[integer_bit]])
            integer_bit += 1    
        
        integer_bit = 0
        for bit in mask_i:
            if bit == '0':
                qc.x(register_qubits[integer_bit])
            integer_bit += 1
        qc.barrier()



rows = 3
cols = 3

arr = np.zeros((rows, cols))

samples = list(np.arange(0, rows*cols))
sample_list = sample(samples, k=rows*cols)
real = random.choice(sample_list)

for i in range(0, rows):
    for j in range(0, cols):
        arr[i, j] = sample_list[i*cols + j]

print(arr)


service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)

n = math.ceil(math.log(rows*cols, 2)) # 4

register_qubits = QuantumRegister(n, name='r') # 4 qbits for register
var_qubits = QuantumRegister(n, name='v') # 4 qbits for value
output_qubit = QuantumRegister(1, name='out') # 1 qbit for output
cbits = ClassicalRegister(n+n, name='cbits') # 8 cbits for register + value
qc = QuantumCircuit(register_qubits, var_qubits, output_qubit, cbits)


qc.initialize([1, -1]/np.sqrt(2), output_qubit)    
qc.h(register_qubits)

qc.barrier()  # for visual separation

forward_qram_n(qc, register_qubits, var_qubits, n, sample_list)

clause = real # 3

oracle_n(qc, clause, var_qubits, n, output_qubit)
qc.barrier()

forward_qram_n(qc, register_qubits, var_qubits, n, sample_list)

qc.barrier()  # for visual separation

qc.append(diffuser(n), register_qubits)

forward_qram_n(qc, register_qubits, var_qubits, n, sample_list)


qc.barrier()
oracle_n(qc, clause, var_qubits, n, output_qubit)
qc.barrier()

forward_qram_n(qc, register_qubits, var_qubits, n, sample_list)

qc.barrier()


qc.append(diffuser(n), register_qubits)
forward_qram_n(qc, register_qubits, var_qubits, n, sample_list)

qc.barrier()
oracle_n(qc, clause, var_qubits, n, output_qubit)
qc.barrier()

forward_qram_n(qc, register_qubits, var_qubits, n, sample_list)

qc.barrier()

qc.append(diffuser(n), register_qubits)

forward_qram_n(qc, register_qubits, var_qubits, n, sample_list)
qc.barrier()
oracle_n(qc, clause, var_qubits, n, output_qubit)
qc.barrier()
forward_qram_n(qc, register_qubits, var_qubits, n, sample_list)

qc.barrier()
qc.append(diffuser(n), register_qubits)
qc.measure(register_qubits, cbits[0:n])
qc.measure(var_qubits, cbits[n:n+n])

qc = qc.reverse_bits()


qc.draw('mpl')

#estimator = Estimator(backend)
#job = estimator.run([qc])
#print(f"job id: {job.job_id()}")
#result = job.result()
#print(result)