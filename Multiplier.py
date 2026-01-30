from typing import Dict, Optional, Sequence
import numpy as np
from default_architecture_class import arhitecture_specs, default_arhitecture
from gates import AndGate, XorGate

def encode_bit_to_phases(bit: float, lambdas: np.ndarray) -> np.ndarray:
    """Encode y∈{0,1} into phases: (y / λ) mod 1."""
    bit = float(bit)
    return (bit / np.asarray(lambdas, dtype=float)) % 1.0

class Multiplier2x2:
    def __init__(self, spec: arhitecture_specs, *, sigma_phase=0.0, sigma_trig=0.0, sigma_score=0.0, p=0.0):

        self.spec = spec

        self.AND1 = AndGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)
        self.AND2 = AndGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)
        self.AND3 = AndGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)
        self.AND4 = AndGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)
        self.AND5 = AndGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)

        self.XOR1 = XorGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)
        self.XOR2 = XorGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)

    def _gate_out(self, gate, in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
        """Run a 2-input gate given two phase vectors of shape (M,)."""
        theta = np.vstack([in1, in2])  # shape (2, M)
        return gate.forward(theta)["out_phases"]
    
    def forward_phases(self, a1: np.ndarray, a0: np.ndarray, b1: np.ndarray, b0: np.ndarray):
        p0 = self._gate_out(self.AND1, a0, b0)   # a0 b0
        t1 = self._gate_out(self.AND2, a1, b0)   # a1 b0
        t2 = self._gate_out(self.AND3, a0, b1)   # a0 b1
        t3 = self._gate_out(self.AND4, a1, b1)   # a1 b1

        # Middle bit + carry
        p1 = self._gate_out(self.XOR1, t1, t2)   # XOR
        c1 = self._gate_out(self.AND5, t1, t2)   # AND = carry from adding t1+t2

        # Upper bits
        p2 = self._gate_out(self.XOR2, t3, c1)
        p3 = self._gate_out(self.AND1, t3, c1)   # reuse AND gate behavior (any AND instance is fine)

        return {"p0": p0, "p1": p1, "p2": p2, "p3": p3}
    
    def forward_bits(self, a1: int, a0: int, b1: int, b0: int):
        """
        Convenience: accept raw bits {0,1}, encode to phases internally.
        """
        lambdas = np.asarray(self.spec.lambdas, dtype=float)
        A1 = encode_bit_to_phases(a1, lambdas)
        A0 = encode_bit_to_phases(a0, lambdas)
        B1 = encode_bit_to_phases(b1, lambdas)
        B0 = encode_bit_to_phases(b0, lambdas)
        return self.forward_phases(A1, A0, B1, B0)