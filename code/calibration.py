import matplotlib.pyplot as plt
import numpy as np

from impl.vis import Plot2DPoints, PlotProjectedPoints, Plot3DPoints, PlotCamera
from impl.calib.geometry import NormalizePoints2D, NormalizePoints3D, EstimateProjectionMatrix, DecomposeP
from impl.calib.io import ReadPoints2D, ReadPoints3D, ReadImageSize
from impl.calib.opt import ImageResiduals, OptimizeProjectionMatrix


def main():
  np.set_printoptions(precision=3)
  # Load the point correspondences
  # Each row is a point. Corresponding points are in the same rows in `points2D` and `points3D`, respectively.
  # 2D points as Nx2 array with [x, y] coordinates
  points2D = ReadPoints2D('../data')
  # 3D points as Nx3 array with [X, Y, Z] coordinates
  points3D = ReadPoints3D('../data')
  # Image size as [width, height]
  image_size = ReadImageSize('../data')

  #Ok, so we know both the 2D and 3D points which will help us callibrate the camera

  # Visualize both 2D and 3D projection, if the camera is the origin then they should look the same. ?
  fig = plt.figure()
  ax3d = fig.add_subplot(121, projection='3d')
  ax2d = fig.add_subplot(122)
  Plot3DPoints(points3D, ax3d)
  Plot2DPoints(points2D, image_size, ax2d)
  
  plt.show(block=False)
  

  # TODO DONE
  # Normalize 2D and 3D points. Normalisation in this case means to that the  mean distance to the origin is 1. Function just takes the points. T2D and T3D is the normalisation matrix.
  # We do this to avoid explosion and dimishing points
  

  normalized_points2D, T2D = NormalizePoints2D(points2D, (19, 2))
  normalized_points3D, T3D = NormalizePoints3D(points3D)
  
  # TODO
  # Estimate the projection matrix from normalized correspondences. P_hat is the estimate of projection matrix. Any additional lines required? Let's check the guidelines
  #This is the DLT part. Where is the contrained matrix. Is this where we need to flattened P and solve linear equations?

  P_hat = EstimateProjectionMatrix(normalized_points2D, normalized_points3D)
  # This runs within the EstimateBuildProjectionConstraintMatrix(normalized_points2D, normalized_points3D)
  #Use SVD to solve this system of equations

  # TODO
  # Optimize based on reprojection error
  P_hat_opt = OptimizeProjectionMatrix(P_hat, normalized_points2D, normalized_points3D)

  print(f'Reprojection error after optimization: {np.linalg.norm(ImageResiduals(P_hat_opt, normalized_points2D, normalized_points3D))**2}')


  # TODO
  # Denormalize P
  # WHICH ONE IS SUPPOSED TO BE INVERTED AND WHY IS NOTHING CHANGING?
  P = np.matmul(np.matmul(T2D, P_hat_opt),T3D)

  # TODO
  # Decompose P
  K, R, t = DecomposeP(P)

  # Print the estimated values
  print(f'K=\n{K/K[2,2]}')
  print(f'R =\n{R}')
  print(f't = {t.transpose()}')

  # Visualize
  PlotCamera(R, t, ax3d, 0.5)
  PlotProjectedPoints(points3D, points2D, K, R, t, image_size, ax2d)

  # Make sure the plots are shown before the program terminates
  plt.show()

if __name__ == "__main__":
  main()