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
            raise ValueError("wrong number of input AND gate")
        
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
            raise ValueError("wrong number of input AND gate")
        
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
            raise ValueError("wrong number of input AND gate")
        
        self.set_logical_weights(list(map(float, logical_weights)))
        x_vals = np.asarray(spec.x_values, dtype=float)
        enc = {}
        for x in x_vals:
            enc[float(x)] = 1.0 if x < 2.0 else 0.0
        encoder_map = enc
        self.set_encoder_map(encoder_map)
        self.compile_encoder_weights()
        