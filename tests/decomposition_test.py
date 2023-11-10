from mola import Matrix
from mola import utils
from mola import decomposition

def test_qr_decomposition():
    """Test QR decomposition."""
    mat = Matrix([ [12, -51, 4], [6, 167, -68], [-4, 24, -41] ])
    Q, R = decomposition.qrd(mat)
    q = Matrix([ [0.8571, -0.3943, 0.3314], [0.4286, 0.9029, -0.0343], [-0.2857, 0.1714, 0.9429]])
    r = Matrix([ [14, 21, -14], [0, 175, -70], [0, 0, -35] ])
    assert(utils.equals_approx(Q,q,1e-4) and utils.equals_approx(R,R,1e-4))
    