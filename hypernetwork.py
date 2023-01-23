import torch
import inspect

# adapted from AUTOs HN code https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/hypernetworks/hypernetwork.py

class HypernetworkModule(torch.nn.Module):
    def __init__(self, dim, layer_structure, activation_func, add_layer_norm, activate_output, dropout_structure):
        super().__init__()
        self.activation_dict = {
            "linear": torch.nn.Identity,
            "relu": torch.nn.ReLU,
            "leakyrelu": torch.nn.LeakyReLU,
            "elu": torch.nn.ELU,
            "swish": torch.nn.Hardswish,
            "tanh": torch.nn.Tanh,
            "sigmoid": torch.nn.Sigmoid,
        }
        self.activation_dict.update({cls_name.lower(): cls_obj for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation) if inspect.isclass(cls_obj) and cls_obj.__module__ == 'torch.nn.modules.activation'})

        linears = []

        for i in range(len(layer_structure) - 1):
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i+1])))

            if activation_func == "linear" or (i >= len(layer_structure) - 2 and not activate_output):
                pass
            elif activation_func in self.activation_dict:
                linears.append(self.activation_dict[activation_func]())

            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i+1])))

            if dropout_structure is not None and dropout_structure[i+1] > 0:
                
                linears.append(torch.nn.Dropout(p=0.0))

        self.linear = torch.nn.Sequential(*linears)
        self.register_buffer("multiplier", torch.tensor(1.0), False)

    def forward(self, x):
        return self.linear(x) * self.multiplier

class Hypernetwork(torch.nn.Module):
    def __init__(self, state_dict) -> None:
        super().__init__()
        self.build_model(state_dict)
        self.load_model(state_dict)

    def parse_dropout_structure(self):
        if self.layer_structure is None:
            self.layer_structure = [1, 2, 1]
        if not self.use_dropout:
            return [0] * len(self.layer_structure)
        dropout_values = [0]
        dropout_values.extend([0.3] * (len(self.layer_structure) - 3))
        if self.last_layer_dropout:
            dropout_values.append(0.3)
        else:
            dropout_values.append(0)
        dropout_values.append(0)
        return dropout_values
    
    def fix_old_state_dict(self, state_dict):
        changes = {
            'linear1.bias': 'linear.0.bias',
            'linear1.weight': 'linear.0.weight',
            'linear2.bias': 'linear.1.bias',
            'linear2.weight': 'linear.1.weight',
        }

        for fr, to in changes.items():
            x = state_dict.get(fr, None)
            if x is None:
                continue

            del state_dict[fr]
            state_dict[to] = x

    def build_model(self, state_dict):
        self.sizes = {320, 640, 768, 1280}.intersection(set(state_dict.keys()))

        self.layer_structure = state_dict.get('layer_structure', [1, 2, 1])
        self.activation_func = state_dict.get('activation_func', "linear")
        self.add_layer_norm = state_dict.get('is_layer_norm', False)
        self.activate_output = state_dict.get('activate_output', True)
        self.dropout_structure = state_dict.get('dropout_structure', None)
        self.last_layer_dropout = state_dict.get('last_layer_dropout', False)
        self.use_dropout = True if self.dropout_structure is not None and any(self.dropout_structure) else state_dict.get('use_dropout', False)

        if self.dropout_structure is None:
            self.dropout_structure = self.parse_dropout_structure()

        self.layers = {}
        for size in self.sizes:
            self.layers[size] = (
                HypernetworkModule(size, self.layer_structure, self.activation_func, self.add_layer_norm, self.activate_output, self.dropout_structure),
                HypernetworkModule(size, self.layer_structure, self.activation_func, self.add_layer_norm, self.activate_output, self.dropout_structure)
            )
            self.add_module(f"{size}_0", self.layers[size][0])
            self.add_module(f"{size}_1", self.layers[size][1])
    
    def load_model(self, state_dict):
        for size in self.sizes:
            self.fix_old_state_dict(state_dict[size][0])
            self.layers[size][0].load_state_dict(state_dict[size][0])

            self.fix_old_state_dict(state_dict[size][1])
            self.layers[size][1].load_state_dict(state_dict[size][1])

    def attach(self, model):
        for name, module in model.modules.items():
            if name.endswith("to_k"):
                suffix = "_0"
            elif name.endswith("to_v"):
                suffix = "_1"
            else:
                continue

            if module.dim in self.sizes:
                hn_module = getattr(self, f"{module.dim}{suffix}")
                model.modules[name].attach_hn(hn_module)

    def set_strength(self, strength):
        for _, module in self.named_modules():
            if hasattr(module, "multiplier"):
                module.multiplier = torch.tensor(strength).to(self.device)

    def __getattr__(self, name):
        if name == "device":
            return next(self.parameters()).device
        return super().__getattr__(name)