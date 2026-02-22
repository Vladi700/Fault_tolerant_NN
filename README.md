# Fault-Tolerant Neural Network (Python)

This repository implements components for **fault-tolerant neural computation** using custom network architectures and logic elements. It provides Python utilities (classes, arithmetic modules, logic-gate based networks) and example notebooks demonstrating experiments with noise/fault injection and neural behavior.

The source code is in Python with several Jupyter notebooks for exploration and testing.

---

## Project Summary

The goal of this project is to explore **neural network architectures and logic circuits** in the presence of noise and faults.  
The code defines:

- Parameterised architecture specifications
- Logic-gate styled network components (AND, OR, XOR, NAND)
- A 2×2 multiplier built from noisy logic modules
- Example notebooks for experiments and testing
- Evaluation on a simple **3-spiral classification problem**

---

## Project Motivation

Traditional neural networks assume reliable computations.  
This project investigates:

- How fault-tolerant neural architectures behave under **noise**
- How to simulate **synaptic failure**
- Whether structured architectures remain robust
- How such models perform on nonlinear classification tasks

---

## Repository Structure

```
Fault_tolerant_NN/
│
├── Multiplier.py                  # 2×2 multiplier built with gates
├── default_architecture_class.py  # Core architecture spec + noisy forward pass
├── gates.py                       # Logic gate classes (And, Or, Xor, Nand, encoder)
│
├── tests.ipynb                    # Tests with example inputs / outputs
├── threeSpiral.py                 # Neural example script (spiral dataset)
├── threeSpiralResults.ipynb       # Notebook with results of spiral experiment
└── .ipynb_checkpoints/            # Notebook checkpoints
```
---

## 🔧 Requirements

This repository uses standard Python and several scientific libraries:

```bash
pip install numpy 
pip install networkx
pip install pytorch
```
---

## 📌 Core Components

### Architecture Specs & Base Class

`default_architecture_class.py` defines:

- `arhitecture_specs`: encapsulates encoding/decoding parameters
- `default_arhitecture`: graph-based computation model with noise and synapse failure support

This base class supports:

- configurable noise models (phase, trig, score)
- synapse drop-out probability (fault injection)
- decoder logic for discrete outputs

You can instantiate this class with different settings and extend it for specific computational modules.

---

### Logic Gates

`gates.py` implements logic gate modules inheriting from the base architecture:

| Class | Description |
|-------|-------------|
| `AndGate` | Outputs logical AND |
| `OrGate` | Outputs logical OR |
| `XorGate` | Outputs logical XOR |
| `NandGate` | Outputs logical NAND |
| `BitPhaseEncoder` | Helper to encode bits into phases |

All gate classes set encoder maps and compile weights into the underlying graph model.  [oai_citation:2‡GitHub](https://raw.githubusercontent.com/Vladi700/Fault_tolerant_NN/main/gates.py)

---

### 2×2 Multiplier

`Multiplier.py` builds a simple 2×2 bit multiplier using:

- four AND gates for partial products
- XOR and AND combinations for sum and carry

It consumes bits as inputs and returns product bits, allowing noise/fault parameters (`sigma_phase`, `sigma_trig`, `sigma_score`, `p`) to be configured when instantiating the class.  [oai_citation:3‡GitHub](https://raw.githubusercontent.com/Vladi700/Fault_tolerant_NN/main/Multiplier.py)

---

## Examples and Notebooks

### `tests.ipynb`

Contains simple tests or experiments validating gate behavior under noise/fault conditions.

### `threeSpiral.py` & `threeSpiralResults.ipynb`

Scripts/notebooks exploring neural patterns (likely the 3-spiral toy dataset), visualization and results. These can be run interactively.

---

## Usage

You can import the modules in Python to simulate noisy neural gates and compositions:

```python
from default_architecture_class import arhitecture_specs, default_arhitecture
from gates import AndGate, XorGate
from Multiplier import Multiplier2x2

# Define an architecture spec
spec = arhitecture_specs(M=4, m0=1, S=2, lambdas=[1.0, 0.5, 0.25, 0.125], x_values=[0.0, 1.0])

# Create an AND gate with fault/noise parameters
and_gate = AndGate(spec, sigma_phase=0.05, sigma_trig=0.01, sigma_score=0.02, p=0.05)

# Forward pass example
result = and_gate.forward([[0.8,0.2,0.5,0.1]])
```

Adapt parameters to simulate different levels of noise/faults across phases, synapses, and scoring.

---

## Extending the Code

This setup is general enough to add:

- Additional gates (e.g., NOR, XNOR)
- Larger digital arithmetic circuits
- Visualization of noise effects on outputs
- Different neural network modules built on the same noisy architecture

---

## Authors
Barış Kalfa,
Vladimir Ungureanu,
Yusuf Kenan Şafak

