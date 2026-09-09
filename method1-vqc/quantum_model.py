import pennylane as qml
from pennylane import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(inputs, weights):

    # Angle encoding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Variational layer
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return qml.expval(qml.PauliZ(0))