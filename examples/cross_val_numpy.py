from ikpls.numpy_ikpls import PLS as NpPLS
import numpy as np

N = 100 # Number of samples
K = 50 # Number of targets
M = 10 # Number of features
A = 20 # Number of latent variables (PLS components)

X = np.random.uniform(size=(N, K), dtype=np.float64)
Y = np.random.uniform(size=(N, M), dtype=np.float64)

# The other PLS algorithms and implementations have the same interface for fit() and predict().
nppls_alg_1 = NpPLS()
nppls_alg_1.fit(X, Y, A)

y_pred = nppls_alg_1.predict(X) # Will have shape (A, N, M)
y_pred_20_components = nppls_alg_1.predict(X, n_components=20) # Will have shape (N, M)