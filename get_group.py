import numpy as np

def remove_outliers(points, factor=1.5):
  """
  Remove outliers from a set of 2D points using the IQR method.
  
  :param points: numpy array of shape (n, 2) where n is the number of points
  :param factor: IQR factor, default is 1.5
  :return: numpy array with outliers removed
  """
  # Primero removemos cualquier fila que contenga NaN
  points = points[~np.isnan(points).any(axis=1)]
  
  # Separate x and y coordinates
  x = points[:, 0]
  y = points[:, 1]
  
  # Calculate Q1, Q3, and IQR for both x and y
  Q1_x, Q3_x = np.percentile(x, [25, 75])
  Q1_y, Q3_y = np.percentile(y, [25, 75])
  IQR_x = Q3_x - Q1_x
  IQR_y = Q3_y - Q1_y
  
  # Calculate the outlier range
  lower_bound_x = Q1_x - factor * IQR_x
  upper_bound_x = Q3_x + factor * IQR_x
  lower_bound_y = Q1_y - factor * IQR_y
  upper_bound_y = Q3_y + factor * IQR_y
  
  # Create a mask for non-outlier points
  mask = ((x >= lower_bound_x) & (x <= upper_bound_x) &
          (y >= lower_bound_y) & (y <= upper_bound_y))
  
  # Return the filtered points
  return points[mask]

def get_roi(person):

    person = remove_outliers(person)

    # reordenar los puntos de la persona de menor a mayor en x
    sort_x = person[np.argsort(person[:, 0])]
    sort_y = person[np.argsort(person[:, 1])]

    x_less = sort_x[0][0] - 50
    y_less = sort_y[0][1] - 50
    x_more = sort_x[-1][0] + 50
    y_more = sort_y[-1][1] + 50

    return x_less, y_less, x_more, y_more

def is_intercept(area1, area2):
    # intercepcion de area
    x1_1, y1_1, x2_1, y2_1 = area1
    x1_2, y1_2, x2_2, y2_2 = area2

    if x1_1 < x2_2 and x2_1 > x1_2 and y1_1 < y2_2 and y2_1 > y1_2:
        return True
    return False
