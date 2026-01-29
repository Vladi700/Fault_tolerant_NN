from default_architecture_class import default_arhitecture
from default_architecture_class import arhitecture_specs

class AndGate(default_arhitecture):
    def __init__(self, spec: arhitecture_specs, a_value=1):
        super().__init__(spec)
        if spec.m0 != 2:
            raise ValueError("wrong number of input AND gate")
        self.set_logical_weights([1, 1])

        x_vals = list(spec.x_values)
        if len(x_vals) != 3:
            raise ValueError("AND gate expects 3 x_values")
        
        x0, xa, x2a = x_vals

        self.set_truth_table({
            x0: x0,
            xa: x0,
            x2a : xa
        })
        #the and gate is not yet complete either