from intel_npu_acceleration_library.backend import MatMul
import numpy as np
import torch
import time
import json

inC, outC = (128, 128)

X1 = np.random.uniform(-1, 1, (1, inC)).astype(np.float16)
X2 = np.random.uniform(-1, 1, (outC, inC)).astype(np.float16)

mm = MatMul(inC, outC, batch=1, profile=True, device="NPU")

result = mm.run(X1, X2)

#print(result)

with open("profiling.json") as fp:
    hwp_runtime = (
        json.load(fp)["taskStatistics"]["total duration"] / 1000.0
    )

print(hwp_runtime)
