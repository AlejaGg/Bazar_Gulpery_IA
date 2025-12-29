"""
Script para cargar modelos YOLO con PyTorch 2.6+ de forma segura
"""

import torch
import warnings

# Monkey-patch torch.load para usar weights_only=False por defecto
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    """torch.load con weights_only=False por defecto para compatibilidad con YOLO"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Aplicar patch
torch.load = patched_torch_load

print("✓ Patch aplicado: torch.load ahora usa weights_only=False por defecto")
