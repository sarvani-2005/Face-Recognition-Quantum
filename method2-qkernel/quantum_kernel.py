import pennylane as qml
from pennylane import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

def feature_map(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
        qml.RZ(x[i], wires=i)

    # Strong entanglement (circular)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    qml.CNOT(wires=[n_qubits-1, 0])


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(n_qubits))


def kernel(x1, x2):
    return abs(kernel_circuit(x1, x2)[0])