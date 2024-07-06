from intel_npu_acceleration_library.backend import MatMul
import numpy as np

inC, outC = (8, 16)

X1 = np.random.uniform(-1, 1, (1, inC)).astype(np.float16)
X2 = np.random.uniform(-1, 1, (outC, inC)).astype(np.float16)

mm = MatMul(inC, outC, batch=1, profile=True)

result = mm.run(X1, X2)

print(result)
