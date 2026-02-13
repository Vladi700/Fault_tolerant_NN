from typing import Dict, Optional, Sequence
import numpy as np
from default_architecture_class import arhitecture_specs, default_arhitecture
from gates import AndGate, XorGate, BitPhaseEncoder
import networkx as nx

class Multiplier2x2:
    def __init__(self, spec: arhitecture_specs, *, sigma_phase=0.0, sigma_trig=0.0, sigma_score=0.0, p=0.0):

        self.spec = spec
        M = spec.M
        lambdas = spec.lambdas

        bit_spec = arhitecture_specs(
    M=M,
    m0=1,
    S=2,  # Only need 2 decoder symbols for bits [0.0, 1.0]
    lambdas=lambdas,
    x_values=np.array([0.0, 1.0], dtype=float)
)
        #Encoder 
        self.enc_a1 = BitPhaseEncoder(bit_spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p, logical_weights=[1])
        self.enc_a0 = BitPhaseEncoder(bit_spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p, logical_weights=[1])
        self.enc_b1 = BitPhaseEncoder(bit_spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p, logical_weights=[1])
        self.enc_b0 = BitPhaseEncoder(bit_spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p, logical_weights=[1])

        # --- Gate 
        self.AND_a0b0 = AndGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)
        self.AND_a1b0 = AndGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)
        self.AND_a0b1 = AndGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)
        self.AND_a1b1 = AndGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)

        self.XOR_p1   = XorGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)
        self.AND_c1   = AndGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)

        self.XOR_p2   = XorGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)
        self.AND_p3   = AndGate(spec, sigma_phase=sigma_phase, sigma_trig=sigma_trig, sigma_score=sigma_score, p=p)

    def _gate(self, gate, x_phase: np.ndarray, y_phase: np.ndarray) -> np.ndarray:
        theta = np.vstack([x_phase, y_phase])       # (2, M)
        return gate.forward(theta)["out_phases"] 
    
    def forward_bits(self, a1: int, a0: int, b1: int, b0: int):
        # --- Encode bits 
        A1 = self.enc_a1.forward_bit(a1)
        A0 = self.enc_a0.forward_bit(a0)
        B1 = self.enc_b1.forward_bit(b1)
        B0 = self.enc_b0.forward_bit(b0)

        # --- Partial products
        p0 = self._gate(self.AND_a0b0, A0, B0)
        t1 = self._gate(self.AND_a1b0, A1, B0)
        t2 = self._gate(self.AND_a0b1, A0, B1)
        t3 = self._gate(self.AND_a1b1, A1, B1)

        # --- Add t1 + t2 (half-adder): p1 + carry c1
        p1 = self._gate(self.XOR_p1, t1, t2)
        c1 = self._gate(self.AND_c1, t1, t2)

        # --- Add t3 + c1 (half-adder): p2 + p3
        p2 = self._gate(self.XOR_p2, t3, c1)
        p3 = self._gate(self.AND_p3, t3, c1)

        return {"p0": p0, "p1": p1, "p2": p2, "p3": p3}
    