import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple

class arhitecture_specs:
    def __init__(self, M, m0, S, lambdas, x_values):
        self.M = M #number of moduli
        self.m0 = m0 #number of inputs from previos layer
        self.S = S #number of decoded values
        self.lambdas = lambdas #moduli
        self.x_values = x_values #possible outputs

class default_arhitecture:
    def __init__(self, spec: arhitecture_specs):
        self.spec = spec
        self.G = nx.DiGraph()
        self._build()

    def graph(self) -> nx.DiGraph:
        return self.G
    
    def set_logical_weights(self, a:List[float]):
        for i in range(self.spec.m0): #index of input
            for j in range(self.spec.M): #index of moduli
                u = f"θ{i}{j}"
                v = f"Φ{j}"
                self.G.edges[u, v]["weight"] = float(a[i])
    
    def set_truth_table(self, map_x_to_y: Dict[float, float]):
        self.G.graph['encoder_map'] = {float(k): float(v) for k, v in map_x_to_y.items()}

    def _build(self):
        M, m0, S = self.spec.M, self.spec.m0, self.spec.S
        #input code space
        for i in range(m0):
            for j in range(M):
                self.G.add_node(
                    f"θ{i}{j}",
                    i=i, j=j,
                    stage="input_code"
                )
        #weighted sum code space
        for j in range(M):
            self.G.add_node(
                f"Φ{j}",
                j=j,
                stage="logical_weights"
            )
        for i in range(m0):
            for j in range(M):
                self.G.add_edge(
                    f"θ{i}{j}" , f"Φ{j}",
                    weight = 1,
                    stage = "logical_weights",
                    name = f"a{i}"
                )
        #decoder neurons(eq.4)
        #sin/cos nodes
        for j in range(M):
            self.G.add_node(
                f"cos{j}",
                j=j,
                stage="trig"
            )
            self.G.add_node(
                f"sin{j}",
                j=j,
                stage="trig"
            )

            self.G.add_edge(f"Φ{j}", f"cos{j}", role = "trig_input", name="2π")
            self.G.add_edge(f"Φ{j}", f"sin{j}", role="trig_input", name="2π")
        
        for k in range(S):
            self.G.add_node(
                f"x{k}",
                k=k,
                x_k=float(self.spec.x_values[k]),
                stage="decoder"
            )

        for j in range(M):
            for k in range(S):
                angle = 2*np.pi * (self.spec.x_values[k] / self.spec.lambdas[j])
                self.G.add_edge(
                    f"cos{j}", f"x{k}",
                    weight=float(np.cos(angle)),
                    role="decode_cos",
                    name = f"Wcos{j}"
                )
                self.G.add_edge(
                    f"sin{j}", f"x{k}",
                    weight=float(np.sin(angle)),
                    role = "decode_sin",
                    name = f"Wsin{j}"
                )
        #Encoder code-space nodes
        for j in range(M):
            self.G.add_node(
                f"Φ'{j}",
                j=j,
                stage="encoder"
            )

        for k in range(S):
            for j in range(M):
                self.G.add_edge(
                    f"x{k}",
                    f"Φ'{j}",
                    W_enc=None,
                    stage = "encoder",
                    name = f"Wenc{j}"   
                )
        self.G.graph["M"] = M
        self.G.graph["m0"] = m0
        self.G.graph["S"] = S
        self.G.graph["lambdas"] = self.spec.lambdas.astype(float)
        self.G.graph["x_values"] = self.spec.x_values.astype(float)
        self.G.graph["encoder_map"] = None






