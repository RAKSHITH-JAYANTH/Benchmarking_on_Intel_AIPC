from intel_npu_acceleration_library.backend import SDPA
import numpy as np
import torch
import time
import json

query_shapes = (1, 8, 16)  
key_shapes = (1, 8, 16)    
value_shapes = (1, 8, 16)   
mask_shapes = (1, 8, 8)    


sdpa_layer = SDPA(
    query_shapes=query_shapes,
    key_shapes=key_shapes,
    value_shapes=value_shapes,
    mask_shapes=mask_shapes,
    is_causal=False,
    profile=True,
    device="NPU"
)

query = np.random.rand(*query_shapes).astype(np.float16)
key = np.random.rand(*key_shapes).astype(np.float16)
value = np.random.rand(*value_shapes).astype(np.float16)
mask = np.random.randint(0, 2, mask_shapes).astype(np.float16)

output = sdpa_layer(query, key, value, mask)
print(output)

with open("profiling.json") as fp:
    hwp_runtime = (
        json.load(fp)["taskStatistics"]["total duration"] / 1000.0
    )
print(hwp_runtime)
