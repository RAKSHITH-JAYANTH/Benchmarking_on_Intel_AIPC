from intel_npu_acceleration_library.backend import MatMul
import numpy as np

inC, outC, batch = (16, 32, 8)

X1 = np.random.uniform(-1, 1, (batch, inC)).astype(np.float16)
X2 = np.random.uniform(-1, 1, (outC, inC)).astype(np.float16)

mm = MatMul(inC, outC, batch, profile=True)

result = mm.run(X1, X2)
