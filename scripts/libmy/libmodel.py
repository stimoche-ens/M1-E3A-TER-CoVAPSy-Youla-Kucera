#!/usr/bin/env python3

import torch
import torch.nn as nn

class SensorPack:
    def __init__(self, sensor_keys, dataset_stats):
        """
        sensor_keys: List of keys e.g. ["speed", "angle", "lidar"]
        dataset_stats: The dictionary containing global size/offset/scale for each key
        """
        self.keys = sensor_keys
        self.stats = dataset_stats
        
        self.total_width = 0
        self.indexes   = {}
        
        self.scale_vec   = []
        self.offset_vec  = []
        
        for key in sensor_keys:
            # e.g., Speed -> size=1, mean=vec(1), std=vec(1), scale=vec(1) aka 1/28
            # e.g., Lidar -> size=360, mean=vec(360), std=vec(360), scale=1/std
            sens_stats = self.stats[key]
            size   = sens_stats["size"]
            width  = size[-1]
            offset = sens_stats["train_offset"]
            scale  = sens_stats["train_scale"]
            
            self.indexes[key] = (self.total_width, self.total_width + width)
            self.total_width += width
            
            self.offset_vec.append(offset)
            self.scale_vec.append(scale)
            
        self.offset_vec  = torch.cat(self.offset_vec)
        self.scale_vec = torch.cat(self.scale_vec)

    def to(self, device):
        self.offset_vec = self.offset_vec.to(device)
        self.scale_vec  = self.scale_vec.to(device)
        return self



class LayerAdapter:
    """Interface for normalizing any layer type."""
    def __init__(self, layer):
        self.layer = layer

    def denorm_input(self, scale, offset):
        raise NotImplementedError

    def denorm_output(self, scale, offset):
        raise NotImplementedError

class LinearAdapter(LayerAdapter):
    """Handles nn.Linear and generic Wx + b layers."""
    def denorm_input(self, scale, offset):
        self.layer.weight /= scale
        # B_new = B_old - (W_new @ Mean)
        # Note: Linear weight is (Out, In), Mean is (In). 
        # Matmul works directly: (Out, In) x (In) -> (Out)
        shift = torch.matmul(self.layer.weight, offset)
        if self.layer.bias is not None:
            self.layer.bias -= shift

    def denorm_output(self, scale, offset):
        self.layer.weight *= scale.unsqueeze(1)
        # B_new = (B_old / Scale) * Std + Mean
        if self.layer.bias is not None:
            self.layer.bias *= scale
            self.layer.bias += offset


class RNNAdapter(LayerAdapter):
    """Handles LSTM, GRU, RNN (Input-to-Hidden weights)."""
    def denorm_input(self, scale, offset):
        self.layer.weight_ih_l0 /= scale
        # B_new = B_old - (W_new @ Mean)
        shift = torch.matmul(self.layer.weight_ih_l0, offset)
        if self.layer.bias_ih_l0 is not None:
            self.layer.bias_ih_l0 -= shift

    def denorm_output(self, scale, offset):
        # RNNs usually don't act as Output layers directly 
        # (they output a hidden state, not a physical quantity).
        raise NotImplementedError("RNNs cannot be automatically denormalized as outputs.")


def get_adapter(layer):
    if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
        return RNNAdapter(layer)
    elif isinstance(layer, nn.Linear):
        return LinearAdapter(layer)
    # Add Conv1dAdapter, Conv2dAdapter here as needed
    else:
        raise ValueError(f"No adapter found for layer type: {type(layer)}")



class NormAwareModule(nn.Module):
    def __init__(self, dataset_stats):
        super().__init__()
        
        if not hasattr(self, 'IO_CONFIG'):
            raise NotImplementedError("Model must define IO_CONFIG")
            
        self._registry = []
        self.dataset_stats = dataset_stats
        
        self.input_packs = [
            SensorPack(keys, dataset_stats) for _, keys in self.IO_CONFIG['inputs']
        ]
        self.output_packs = [
            SensorPack(keys, dataset_stats) for _, keys in self.IO_CONFIG['outputs']
        ]

    def _register(self, layer, pack, mode):
        """Internal helper to attach adapters."""
        adapter = get_adapter(layer)
        self._registry.append((adapter, pack, mode))

    def build_input_layer(self, input_idx, LayerClass, **kwargs):
        """
        Factory: Creates a layer attached to a specific Input Config.
        It AUTOMATICALLY sets 'input_size' (or equivalent) based on the Pack.
        """
        pack = self.input_packs[input_idx]
        
        if issubclass(LayerClass, (nn.RNNBase, nn.LSTM, nn.GRU)):
            kwargs['input_size'] = pack.total_width
        elif issubclass(LayerClass, nn.Linear):
            kwargs['in_features'] = pack.total_width
        # ... add Conv logic here ...
            
        layer = LayerClass(**kwargs)
        self._register(layer, pack, mode="input")
        return layer

    def build_output_layer(self, output_idx, LayerClass, **kwargs):
        """
        Factory: Creates a layer attached to a specific Output Config.
        It AUTOMATICALLY sets 'out_features' (or equivalent) based on the Pack.
        """
        pack = self.output_packs[output_idx]
        
        if issubclass(LayerClass, nn.Linear):
            kwargs['out_features'] = pack.total_width
            
        layer = LayerClass(**kwargs)
        self._register(layer, pack, mode="output")
        return layer

    def denormalize_weights(self):
        print("🔧 Adapter-Based Weight Patching...")
        with torch.no_grad():
            for adapter, pack, mode in self._registry:
                if isinstance(adapter.layer, nn.Module):
                    try:
                        # Grab the device of the first parameter (weight/bias)
                        device = next(adapter.layer.parameters()).device
                    except StopIteration:
                        try:
                            device = next(adapter.layer.buffers()).device
                        except StopIteration:
                             raise RuntimeError(f"Layer {adapter.layer} has no parameters or buffers to infer device.")
                else:
                    raise TypeError(
                        f"Adapter layer must be an nn.Module, but got '{type(adapter.layer)}'. "
                        "If this is a custom object, it must expose .parameters()."
                    )

                scale  = pack.scale_vec.to(device)
                offset = pack.offset_vec.to(device)

                if mode == "input":
                    adapter.denorm_input(scale, offset)
                elif mode == "output":
                    adapter.denorm_output(scale, offset)
        print("✅ Done.")

