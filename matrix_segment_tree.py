"""Segment tree helpers for ordered matrix range products."""

import numpy as np
from scipy.sparse import eye, isspmatrix, isspmatrix_csr


def _copy_matrix(matrix):
    if hasattr(matrix, "copy"):
        return matrix.copy()
    return np.array(matrix, copy=True)


class OrderedMatrixProductSegmentTree:
    """Segment tree for products of ordered square matrices."""

    def __init__(self, matrices, force_csr=False):
        if len(matrices) == 0:
            raise ValueError("matrices must be non-empty")

        self.num_matrices = len(matrices)
        self.force_csr = force_csr
        self.shape = matrices[0].shape
        if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
            raise ValueError("matrices must be square")

        self.dtype = getattr(matrices[0], "dtype", np.float64)
        self._is_sparse = isspmatrix(matrices[0])
        self.leaf_count = 1 << (self.num_matrices - 1).bit_length()
        self.nodes = [None] * (2 * self.leaf_count)
        identity = self._make_identity()

        for idx, matrix in enumerate(matrices):
            self.nodes[self.leaf_count + idx] = self._prepare_leaf(matrix)
        for idx in range(self.num_matrices, self.leaf_count):
            self.nodes[self.leaf_count + idx] = identity

        for idx in range(self.leaf_count - 1, 0, -1):
            self.nodes[idx] = self._matmul(self.nodes[2 * idx], self.nodes[2 * idx + 1])

    def _make_identity(self):
        n = self.shape[0]
        if self._is_sparse:
            return eye(n, dtype=self.dtype, format="csr")
        return np.eye(n, dtype=self.dtype)

    def _prepare_leaf(self, matrix):
        if matrix.shape != self.shape:
            raise ValueError("all matrices must have the same shape")
        if self.force_csr and isspmatrix(matrix) and not isspmatrix_csr(matrix):
            matrix = matrix.tocsr()
        return _copy_matrix(matrix)

    def _matmul(self, left, right):
        product = left @ right
        if self.force_csr and isspmatrix(product) and not isspmatrix_csr(product):
            product = product.tocsr()
        return product

    def query(self, left, right):
        """Return the ordered product T[left] @ ... @ T[right]."""
        if left < 0 or right < left or right >= self.num_matrices:
            raise IndexError("invalid query range")

        left_product = None
        right_product = None
        lo = left + self.leaf_count
        hi = right + self.leaf_count + 1

        while lo < hi:
            if lo & 1:
                left_product = (
                    self.nodes[lo]
                    if left_product is None
                    else self._matmul(left_product, self.nodes[lo])
                )
                lo += 1

            if hi & 1:
                hi -= 1
                right_product = (
                    self.nodes[hi]
                    if right_product is None
                    else self._matmul(self.nodes[hi], right_product)
                )

            lo >>= 1
            hi >>= 1

        if left_product is None:
            result = right_product
        elif right_product is None:
            result = left_product
        else:
            result = self._matmul(left_product, right_product)

        return _copy_matrix(result)
