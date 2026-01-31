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
    def __init__(self, spec: arhitecture_specs, *, sigma_phase=0, sigma_trig=0, sigma_score=0, p=0):
        self.spec = spec
        self.G = nx.DiGraph()

        #noise
        self.rng = np.random.default_rng()
        self.sigma_phase = float(sigma_phase)
        self.sigma_trig  = float(sigma_trig)
        self.sigma_score = float(sigma_score)

        #synaptic failure
        self.p = float(p)
        self._build()
    
    def _mod1(self, x):
        return x - np.floor(x)
    
    def _add_phase_noise(self, phases):
        if self.sigma_phase <= 0:
            return phases
        noisy = phases + self.rng.normal(0.0, self.sigma_phase, size=np.shape(phases))
        return self._mod1(noisy)
    
    def _add_gauss(self, x, sigma):
        if sigma <= 0:
            return x
        return x + self.rng.normal(0.0, sigma, size=np.shape(x))
    
    def _syn_mask(self):
        """Bernoulli(1-p_syn): 1 means synapse works, 0 means fails."""
        if self.p <= 0:
            return 1.0
        return 0.0 if (self.rng.random() < self.p) else 1.0

    def graph(self) -> nx.DiGraph:
        return self.G
    
    
    def set_logical_weights(self, a:List[float]):
        for i in range(self.spec.m0): #index of input
            for j in range(self.spec.M): #index of moduli
                u = f"θ{i}{j}"
                v = f"Φ{j}"
                self.G.edges[u, v]["weight"] = float(a[i])

    def set_encoder_map(self, map_x_to_y: Dict[float, float]):
        """
        Store the encoder "truth-table" mapping x_k -> y.
        This is the key thing that differs between gates.
        """
        self.G.graph["encoder_map"] = {float(k): float(v) for k, v in map_x_to_y.items()}
    
    def compile_encoder_weights(self):
        enc_map = self.G.graph.get("encoder_map", None)
        if enc_map is None:
            raise ValueError("No encoder map")
        lambdas = np.asarray(self.spec.lambdas, dtype=float)
        x_vals  = np.asarray(self.spec.x_values, dtype=float)
        for k, xk in  enumerate(x_vals):
            keys = np.array(list(enc_map.keys()), dtype=float)
            nearest = float(keys[np.argmin(np.abs(keys - xk))])
            y = float(enc_map[nearest])
            phases = (y / lambdas) % 1.0
            for j in range(self.spec.M):
                u = f"x{k}"
                v = f"Φ'{j}"   
                self.G.edges[u, v]["weight"] = float(phases[j])

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

            self.G.add_edge(f"Φ{j}", f"cos{j}", role = "trig_input", name="2π", weight= 2 * np.pi)
            self.G.add_edge(f"Φ{j}", f"sin{j}", role="trig_input", name="2π",  weight= 2 * np.pi)
        
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
                    weight=1,
                    stage = "encoder",
                    name = f"Wenc{j}"   
                )
        self.G.graph["M"] = M
        self.G.graph["m0"] = m0
        self.G.graph["S"] = S
        self.G.graph["lambdas"] = self.spec.lambdas.astype(float)
        self.G.graph["x_values"] = self.spec.x_values.astype(float)
        self.G.graph["encoder_map"] = None
    
    
    def _encode(self, y:float):
        y = float(y)
        return self._mod1(y / self.spec.lambdas.astype(float))
    
    def forward(self, theta:np.ndarray):
        M, m0, S = self.spec.M, self.spec.m0, self.spec.S
        theta = np.asarray(theta, dtype=float)
        assert theta.shape == (m0, M), f"theta must have shape ({m0},{M})"

        theta = self._add_phase_noise(theta)

        #input nodes
        for i in range(m0):
            for j in range(M):
                self.G.nodes[f"θ{i}{j}"]["value"] = float(theta[i, j])
        
        #compute phi_j = sum_i w_i * theta[i,j]
        phi = np.zeros(M)
        for j in range(M):
            acc = 0
            for i in range(m0):
                u = f"θ{i}{j}"
                v = f"Φ{j}"
                w = self.G.edges[u, v].get("weight")
                mask = self._syn_mask()
                acc += (mask * w) * float(self.G.nodes[u]["value"])
            phi[j] = self._mod1(acc)

        phi = self._add_phase_noise(phi)

        for j in range(M):
            self.G.nodes[f"Φ{j}"]["value"] = float(phi[j])

        # trig expansion
        cos_phi = np.cos(2 * np.pi * phi)
        sin_phi = np.sin(2 * np.pi * phi)
        cos_phi = self._add_gauss(cos_phi, self.sigma_trig)
        sin_phi = self._add_gauss(sin_phi, self.sigma_trig)
        for j in range(M):
            self.G.nodes[f"cos{j}"]["value"] = float(cos_phi[j])
            self.G.nodes[f"sin{j}"]["value"] = float(sin_phi[j])

        #decoder
        scores = np.zeros(S, dtype=float)
        for k in range(S):
            node_k = f"x{k}"
            acc = 0.0
            #cos
            for j in range(M):
                u = f"cos{j}"
                if self.G.has_edge(u, node_k):
                    w = float(self.G.edges[u, node_k].get("weight"))
                    mask = self._syn_mask()
                    acc += (mask * w) * float(self.G.nodes[u]["value"])
            
            #sin
                u = f"sin{j}"
                if self.G.has_edge(u, node_k):
                    w = float(self.G.edges[u, node_k].get("weight"))
                    mask = self._syn_mask()
                    acc += (mask * w) * float(self.G.nodes[u]["value"])
            
            scores[k] = acc

        scores = self._add_gauss(scores, self.sigma_score)

        for k in range(S):
            self.G.nodes[f"x{k}"]["value"] = float(scores[k])

        k_hat = int(np.argmax(scores))

        s = np.zeros(S, dtype=float)
        s[k_hat] = 1.0
        for k in range(S):
            self.G.nodes[f"x{k}"]["sel"] = float(s[k])

        out_phases = np.zeros(M, dtype=float)
        for j in range(M):
            acc = 0.0
            v = f"Φ'{j}"

            for k in range(S):
                u = f"x{k}"
                w = self.G.edges[u, v].get("weight")

                mask = self._syn_mask()
                acc += mask * float(w) * float(s[k])

            out_phases[j] = acc

        #output
        out_phases = self._add_phase_noise(out_phases)
        out_phases = self._mod1(out_phases)

        # store in encoder nodes (adjust name if yours differs)
        for j in range(M):
            self.G.nodes[f"Φ'{j}"]["value"] = float(out_phases[j])

            return {
                "theta_noisy": theta,
                "phi": phi,
                "trig_cos": cos_phi,
                "trig_sin": sin_phi,
                "scores": scores,
                "k_hat": k_hat,
                "out_phases": out_phases,
            }

            

 
    







