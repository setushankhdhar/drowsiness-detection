
import numpy as np

def eye_aspect_ratio(mesh_points, eye_indices):
    # Extract eye landmarks
    eye = np.array([mesh_points[p] for p in eye_indices])

    # Compute the vertical distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute EAR
    ear = (A + B) / (2.0 * C)
    return ear
