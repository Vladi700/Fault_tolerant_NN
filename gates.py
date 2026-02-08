from typing import Dict, Optional, Sequence
from default_architecture_class import default_arhitecture
from default_architecture_class import arhitecture_specs
import numpy as np

class AndGate(default_arhitecture):
    def __init__(self, spec: arhitecture_specs, *, sigma_phase: float, sigma_trig: float, sigma_score: float, p: float, logical_weights = [1, 1], encoder_map: Optional[Dict[float, float]] = None):
        super().__init__(spec,
                         sigma_phase=sigma_phase,
                         sigma_trig=sigma_trig,
                         sigma_score=sigma_score,
                         p=p)
        if spec.m0 != 2:
            raise ValueError("wrong number of input AND gate")
        
        self.set_logical_weights(list(map(float, logical_weights)))
        x_vals = np.asarray(spec.x_values, dtype=float)
        enc = {}
        for x in x_vals:
            enc[float(x)] = 1.0 if np.isclose(x, 2.0) else 0.0
        encoder_map = enc
        self.set_encoder_map(encoder_map)
        self.compile_encoder_weights()

class OrGate(default_arhitecture):
    def __init__(self, spec: arhitecture_specs, *, sigma_phase: float, sigma_trig: float, sigma_score: float, p: float, logical_weights = [1, 1], encoder_map: Optional[Dict[float, float]] = None):
        super().__init__(spec,
                         sigma_phase=sigma_phase,
                         sigma_trig=sigma_trig,
                         sigma_score=sigma_score,
                         p=p)
        if spec.m0 != 2:
            raise ValueError("wrong number of input OR gate")
        
        self.set_logical_weights(list(map(float, logical_weights)))
        x_vals = np.asarray(spec.x_values, dtype=float)
        enc = {}
        for x in x_vals:
            enc[float(x)] = 1.0 if x > 0.5 else 0.0
        encoder_map = enc
        self.set_encoder_map(encoder_map)
        self.compile_encoder_weights()

class XorGate(default_arhitecture):
    def __init__(self, spec: arhitecture_specs, *, sigma_phase: float, sigma_trig: float, sigma_score: float, p: float, logical_weights = [1, 1], encoder_map: Optional[Dict[float, float]] = None):
        super().__init__(spec,
                         sigma_phase=sigma_phase,
                         sigma_trig=sigma_trig,
                         sigma_score=sigma_score,
                         p=p)
        if spec.m0 != 2:
            raise ValueError("wrong number of input XOR gate")
        
        self.set_logical_weights(list(map(float, logical_weights)))
        x_vals = np.asarray(spec.x_values, dtype=float)
        enc = {}
        for x in x_vals:
            enc[float(x)] = 1.0 if abs(x - 1.0) < 1e-6 else 0.0
        encoder_map = enc
        self.set_encoder_map(encoder_map)
        self.compile_encoder_weights()

class NandGate(default_arhitecture):
    def __init__(self, spec: arhitecture_specs, *, sigma_phase: float, sigma_trig: float, sigma_score: float, p: float, logical_weights = [1, 1], encoder_map: Optional[Dict[float, float]] = None):
        super().__init__(spec,
                         sigma_phase=sigma_phase,
                         sigma_trig=sigma_trig,
                         sigma_score=sigma_score,
                         p=p)
        if spec.m0 != 2:
            raise ValueError("wrong number of input NAND gate")
        
        self.set_logical_weights(list(map(float, logical_weights)))
        x_vals = np.asarray(spec.x_values, dtype=float)
        enc = {}
        for x in x_vals:
            enc[float(x)] = 1.0 if x < 2.0 else 0.0
        encoder_map = enc
        self.set_encoder_map(encoder_map)
        self.compile_encoder_weights()

class BitPhaseEncoder(default_arhitecture):
    def __init__(self, M, lambdas, *, sigma_phase=0.0, sigma_trig=0.0, sigma_score=0.0, p=0.0):
        spec = arhitecture_specs(
        M=M,
        m0=1,                      # one upstream signal
        S=2,                       # two symbols
        lambdas=np.asarray(lambdas, float),
        x_values=np.array([0.0, 1.0], dtype=float),
    )
        super().__init__(spec,
                         sigma_phase=sigma_phase,
                         sigma_trig=sigma_trig,
                         sigma_score=sigma_score,
                         p=p)
        
        self.set_logical_weights([1.0])
        self.set_encoder_map({0.0: 0.0, 1.0: 1.0})
        self.compile_encoder_weights()

    def forward_bit(self, bit: int):
        bit = int(bit)
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        
        theta = self._encode(bit)[None, :]   
        return self.forward(theta)["out_phases"]
        