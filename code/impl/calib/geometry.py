from re import M
import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import HNormalize

def NormalizePoints3D(points):
  
  # Compute the center and spread of points
  center = np.mean(points, 0)
  offsets = points - np.tile(center, (points.shape[0], 1))
  dists = np.linalg.norm(offsets, axis=1)

  T_inv = np.eye(4) * np.mean(dists)
  T_inv[3,3] = 1
  T_inv[0:3,3] = center

  # Invert this so that after the transformation, the points are centered and their mean distance to the origin is 1
  T = np.linalg.inv(T_inv)

  # Normalize the points
  normalized_points3D = (T @ np.append(points, np.ones((points.shape[0], 1)), 1).transpose()).transpose()

  return normalized_points3D[:,0:3], T


def NormalizePoints2D(points, image_size):
  # Assume the image spans the range [-1, 1] in both dimensions and normalize the points accordingly
  T_inv = np.eye(3)
  T_inv[0,0] = image_size[0] / 2
  T_inv[1,1] = image_size[1] / 2
  T_inv[0,2] = image_size[0] / 2
  T_inv[1,2] = image_size[1] / 2

  T = np.linalg.inv(T_inv)

  normalized_points2D = (T @ np.append(points, np.ones((points.shape[0], 1)), 1).transpose()).transpose()

  return normalized_points2D[:,0:2], T


def EstimateProjectionMatrix(points2D, points3D):
  # TODO Build constraint matrix
  # Hint: Pay attention to the assumed order of the vectorized P matrix. 
  # You will need the same order when reshaping the vector to the matrix later
  constraint_matrix = BuildProjectionConstraintMatrix(points2D, points3D)

  # Solve for the nullspace
  print('Constraint Matrix')
  print(constraint_matrix.shape)
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  
  # TODO: Reshape the vector to a matrix (pay attention to the order)

  
  P = np.vstack((P_vec[0:4].transpose(), P_vec[4:8].transpose()))
  
  P = np.vstack((P, P_vec[8:12].transpose()))
  
  return P


def DecomposeP(P):
  # TODO
  # Decompose P into K, R, and t

  # P = K[R|t] = K[R | -RC] = [KR | -KRC]
  # We could decompose KR with a RQ decomposition since K is upper triangular and R is orthogonal
  # To switch this around we set M = KR -> M^(-1) = R^(-1) K^(-1) and can use the QR decomposition on M^(-1)
  

  """   Yes indeed, I discovered the same. In fact, as far as I understood, the q matrix is the orthonormal one and the r is the triangular one and in the notes they do:

    K_inv, R_inv = qr(M_inv) but I think it should be either K_inv, R_inv = rq(M_inv) or R_inv, K_inv = qr(M_inv).
    I don't know if you agree but this is what solved everything for me. At least, I had the same form as in the hand out and it did not stick to the documentation of numpy for the qr decomposition.

    Just as a question, for you too the red points are slightly off the black ones ? """

  'WHY IS P SO MASSIVE ALREADY HERE?'  
  M = P[:, :3]

  M_inv = np.linalg.inv(M)
  R_inv, K_inv = np.linalg.qr(M_inv)

  # TODO
  # Find K and R
  K = np.linalg.inv(K_inv)
  R = np.linalg.inv(R_inv)


  # TODO
  # It is possible that a sign was assigned to the wrong matrix during decomposition
  # We need to make sure that det(R) = 1 to have a proper rotation
  # We also want K to have a positive diagonal
  determinant = np.linalg.det(R)
  
  if int(determinant)== -1:
    R = -R


  T = np.diag(np.sign(np.diag(K)))
  K = np.matmul(K, T) # Is this how we do it?


  R = np.matmul(np.linalg.inv(T), R)

  # TODO
  # Find the camera center C as the nullspace of P
  
  rcond=None
  u, s, vh = np.linalg.svd(P, full_matrices=True)
  m, n = u.shape[0], vh.shape[1]

  print('Hermitian Matrix')
  print(vh)
  if rcond is None:
      rcond = np.finfo(s.dtype).eps * max(m, n)

  tol = np.amax(s) * rcond
  num = np.sum(s > tol)
  C = vh[num:,:].T.conj()


  C = HNormalize(C)

  print(C)
  print('C shape', C.shape)
  # TODO
  # Compute t from R and C
  t = np.matmul(-R, C) 

  return K, R, t