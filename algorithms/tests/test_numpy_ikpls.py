from sklearn.cross_decomposition import PLSRegression

def test_check_stuff():
    sklearn_pls = PLSRegression(n_components=10, scale=False)
    from algorithms.numpy_ikpls import PLS
    numpy_ikpls = PLS(algorithm=1)
    assert 5 != 6