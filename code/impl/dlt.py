import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))

  points3D = np.append(points3D, np.ones((19,1)), axis=1)

  for i in range(num_corrs):
    # TODO Add your code here. BUILD TWO ROWS OF THE CONSTRAINT MATRIX AND STACK THEM. OK HOW DOES ONE DO THIS. CHECK ASSIGNMENT
    # a * vec(P) = 0
    # To do this efficiently, we are going to use SVD on the output of this function
    #Top Row
    
    
    #-w 
    constraint_matrix[2*i][4:8] = -1*points3D[i]
    # y
    constraint_matrix[2*i][8:12] = points2D[i][1]*points3D[i]

    #Bottom Row
    #
    constraint_matrix[2*i + 1][0:4] = points3D[i]
    constraint_matrix[2*i + 1][8:12] = points2D[i][0]*points3D[i]

  return constraint_matrix