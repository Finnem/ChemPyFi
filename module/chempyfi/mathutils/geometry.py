
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
#from numba import jit

def sample_cone(directions, radius, max_angle, ideal_spacing):
    """ Generates point systematically on a cone around the given direction. The cone is defined by the radius and the maximum angle.
    The distance between the points is approximately ideal_spacing.

    Args:
        direction (np.ndarray): 3D vector.
        radius (float): Radius of the cone.
        max_angle (float): Maximum angle of the cone in radians.
        ideal_spacing (float): Ideal spacing between the points.
    """

    # first we generate points along the z axis
    # we first identify rings along which to generate the points
    n_theta = int(np.round(max_angle / np.arccos(1 - ((ideal_spacing / radius)**2 / 2))))
    thetas = np.linspace(0, max_angle, n_theta)

    # we then generate points along the rings
    n_phis = np.round(2 * np.pi * np.sin(thetas) / np.arccos(1 - ((ideal_spacing / radius)**2 / 2)))
    np.clip(n_phis, 1, None, out=n_phis)
    generated_points = []
    for n_phi, theta in zip(n_phis, thetas):
        phi = np.linspace(0, 2 * np.pi, int(n_phi))
        points = np.zeros((int(n_phi), 3))
        points[:, 0] = radius * np.sin(theta).repeat(n_phi) * np.cos(phi)
        points[:, 1] = radius * np.sin(theta).repeat(n_phi) * np.sin(phi)
        points[:, 2] = radius * np.cos(theta).repeat(n_phi)
        generated_points.append(points)
    points = np.concatenate(generated_points, axis=0)

    # we then rotate the points to the correct direction
    directions = np.array(directions)
    if len(directions.shape) == 2:
        return np.array([np.dot(align_directions([0, 0, 1], direction), points.T).T for direction in directions])
    return np.dot(align_directions([0, 0, 1], directions), points.T).T

#@jit
def align_directions(from_direction, to_direction):
    """Find the rotation that aligns from_direction to to_direction.

    Args:
        from_direction (np.ndarray): 3D vector.
        to_direction (np.ndarray): 3D vector.

    Returns:
        np.ndarray: Rotation matrix.
    
    """
    from_direction = from_direction / np.linalg.norm(from_direction)
    to_direction = to_direction / np.linalg.norm(to_direction)
    if np.allclose(from_direction, to_direction):
        return np.eye(3)
    elif np.allclose(from_direction, -to_direction):
        rotation_axis = get_perp(from_direction)
    else:
        rotation_axis = np.cross(from_direction, to_direction)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.dot(from_direction, to_direction))
    rotation_matrix = R.from_rotvec(rotation_angle * rotation_axis).as_matrix()
    return rotation_matrix

def align_to_axes(from_origin, from_axis, from_plane, target_origin = [0, 0, 0], target_axis = "x", target_plane = "z"):
    """Find the rotation and t*ranslation that aligns the given axis and plane to the target axis and plane.

    Args:
        from_origin (np.ndarray): 3D vector. Point which should be translated into the origin.
        from_axis (np.ndarray): 3D vector. Point which should be translated to the axis.
        from_plane (np.ndarray): 3D vector. Point which should be translated to the plane.
        target_origin (np.ndarray): 3D vector. Target origin.
        target_axis (np.ndarray): 3D vector. Target axis. Should be the vector that points into the direction of the target axis.
        target_plane (np.ndarray): 3D vector. Axis which together with the target axis spans the target plane.

    Returns:
        tuple(np.ndarray, np.ndarray): Rotation matrix, translation vector.
    
    """
    from_axis_direction = (from_axis - np.array(from_origin)); from_axis_direction /= np.linalg.norm(from_axis_direction)
    from_plane_direction = (from_plane - np.array(from_origin)); from_plane_direction /= np.linalg.norm(from_plane_direction)
    

    if target_axis in ["x", "y", "z"]:
        actual_target_axis = np.array([0, 0, 0])
        actual_target_axis["xyz".index(target_axis)] = 1
        target_axis = actual_target_axis
    else:
        target_axis = np.array(target_axis); target_axis = target_axis / np.linalg.norm(target_axis)

    if target_plane in ["x", "y", "z"]:
        actual_target_plane = np.array([0, 0, 0])
        actual_target_plane["xyz".index(target_plane)] = 1
        target_plane = actual_target_plane
    else:
        target_plane = np.array(target_plane); target_plane = target_plane / np.linalg.norm(target_plane)
    target_plane = target_plane - np.dot(target_plane, target_axis) * target_axis
    first_rotation = align_directions(from_axis_direction, target_axis)
    from_plane_direction = first_rotation @ from_plane_direction
    target_direction = from_plane_direction - (from_plane_direction @ target_axis * target_axis)
    if np.isclose(target_direction @ target_plane, -np.linalg.norm(target_direction)):
        second_rotation = R.from_rotvec(np.pi * target_axis).as_matrix()
    else:
        second_rotation = align_directions(target_direction, target_plane)
    rotation = second_rotation @ first_rotation
    translation = np.array(target_origin) - rotation @ np.array(from_origin)

    return rotation, translation




def get_perp(v):
    """Return a vector perpendicular to the given vector.
    Will return the vector [0, 1, 0] if v is parallel to [0, 0, 1] or v is the zero vector.
    
    Args:
        v (np.ndarray): 3D vector.
        
    Returns:
        np.ndarray: 3D vector perpendicular to the given vector.
            
    """
    if np.allclose(v, 0):
        return np.array([0, 1, 0])
    v = v / np.linalg.norm(v)
    if np.allclose(np.abs(v), [0, 0, 1]):
        return np.array([0, 1, 0])
    cross = np.cross(v, [0, 0, 1])
    return cross / np.linalg.norm(cross)


def transform(points, rotation, translation):
    """Apply a rotation and translation to the given points.
    
    Args:
        points (np.ndarray): Nx3 matrix of N 3D points.
        rotation (np.ndarray): 3x3 rotation matrix.
        translation (np.ndarray): 3D translation vector.
    
    """
    return (rotation @ points.T + translation[:, None]).T

def apply_transform(points, rotation, translation):
    """ Compatibility function for transform.

    Args:
        points (np.ndarray): Nx3 matrix of N 3D points.
        rotation (np.ndarray): 3x3 rotation matrix.
        translation (np.ndarray): 3D translation vector.
    
    """
    return transform(points, rotation, translation)

def rigid_transform(A, B, sanity_check_tolerance=None):
    """Find the rigid transformation that aligns A to B.

    Args:
        A (np.ndarray): Nx3 matrix of N 3D points.
        B (np.ndarray): Nx3 matrix of N 3D points.
        sanity_check_tolerance (float, optional): Maximum allowable deviation for sanity check.

    Returns:
        tuple(np.ndarray, np.ndarray): Rotation matrix (3x3), translation vector (3,).
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A has shape {A.shape}, B has shape {B.shape}")

    if A.shape[1] != 3:
        raise ValueError(f"Input matrices must have shape Nx3, but got {A.shape}")

    A, B = A.T, B.T  # Transpose to 3xN for calculations

    # Sanity check: Ensure pairwise distances remain consistent
    if sanity_check_tolerance is not None:
        pairwise_distances_A = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)
        pairwise_distances_B = np.linalg.norm(B[:, None, :] - B[None, :, :], axis=-1)
        max_deviation = np.max(np.abs(pairwise_distances_A - pairwise_distances_B))
        if max_deviation > sanity_check_tolerance:
            print(f"Pairwise distances deviate by {max_deviation:.6f}. Possible misalignment in points.")

    # Compute centroids
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # Subtract centroids
    Am, Bm = A - centroid_A, B - centroid_B

    # Compute covariance matrix
    H = Am @ Bm.T

    # Compute SVD
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        #logging.warning("det(R) < 0, reflection detected! Correcting...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Compute translation vector
    t = centroid_B - R @ centroid_A

    return R, t.flatten()

    
def sample_solvent_accessible_surface(atom_coordinates, atom_radii, distance_radius, point_distance, smooth_distance = 3):
    """ Sample the solvent accessible surface of the given atoms. Not intended for use with large molecules.
    
    Args:
        atom_coordinates (np.ndarray): Nx3 matrix of atom coordinates.
        atom_radii (np.ndarray): N vector of atom radii.
        distance_radius (float): Radius of the sphere around each atom.
        point_distance (float): Distance between the points on the sphere.
        smooth_distance (float): Distance for atoms to be considered for smoothing. If 0 no smoothing is applied.

    Returns:
        np.ndarray: Nx3 matrix of points on the solvent excluded surface.
    """
    
    atom_coordinates = np.array(atom_coordinates)
    atom_radii = np.array(atom_radii)

    # sample points around each atom
    points = []
    directions = []

    #def sample_cone(directions, radius, max_angle, ideal_spacing):
    for i, atom_coordinate, atom_radius in zip(np.arange(len(atom_coordinates)), atom_coordinates, atom_radii):
        new_points = sample_cone([0, 0, 1], atom_radius + distance_radius, np.pi, 2 * point_distance)
        # translate points to the correct position
        new_points += atom_coordinate
        # eliminate points inside other atoms
        atom_mask = np.full(len(atom_coordinates), True); atom_mask[i] = False
        pairwise_distances  = np.linalg.norm(new_points[:, None, :] - atom_coordinates[None, atom_mask, :], axis=-1)
        new_points = new_points[np.all(pairwise_distances > atom_radii[atom_mask] + distance_radius, axis=1)]
        points.append(new_points)
        
        # Only compute directions if we have points
        if len(new_points) > 0:
            # Calculate outward-pointing normals: from atom center to surface points
            direction = new_points - atom_coordinate
            direction_norms = np.linalg.norm(direction, axis=1)
            # Avoid division by zero by only normalizing non-zero vectors
            valid_directions = direction_norms > 1e-10
            if np.any(valid_directions):
                direction[valid_directions] /= direction_norms[valid_directions, None]
                directions.append(direction)
            else:
                # If no valid directions, create empty array with correct shape
                directions.append(np.empty((0, 3)))
        else:
            # If no points, create empty array with correct shape
            directions.append(np.empty((0, 3)))
    points = np.concatenate(points, axis=0)
    single_directions = np.concatenate(directions, axis=0)
    kdtree = cKDTree(atom_coordinates)

    # smooth directions locally
    if smooth_distance != 0:
        neighbors = kdtree.query_ball_point(points, smooth_distance)
        #print(neighbors)
        directions = []
        for i, neighbor in enumerate(neighbors):
            if len(neighbor) == 0:
                # If no neighbors, use the original direction
                directions.append(single_directions[i])
                continue
                
            # Calculate outward-pointing directions from atom centers to surface point
            differences = points[i] - atom_coordinates[neighbor]
            distances = np.linalg.norm(differences, axis=1)
            
            # Only consider non-zero distances
            valid_distances = distances > 1e-10
            if not np.any(valid_distances):
                # If all distances are zero, use the original direction
                directions.append(single_directions[i])
                continue
                
            weights = np.exp(-distances[valid_distances] / smooth_distance) + 1
            weights /= np.sum(weights)
            neighbor_directions = differences[valid_distances]
            neighbor_directions /= distances[valid_distances, None]
            directions.append(np.sum(weights[:, None] * neighbor_directions, axis=0))
        directions = np.array(directions)
        
        # Normalize directions, avoiding division by zero
        direction_norms = np.linalg.norm(directions, axis=1)
        valid_directions = direction_norms > 1e-10
        if np.any(valid_directions):
            directions[valid_directions] /= direction_norms[valid_directions, None]
    else:
        directions = single_directions

    return points, directions#, point_colors







def sample_heart(direction, radius, max_angle, ideal_spacing):
    """ Romantically impress someone with my faulty implementation of sample_cone.

    Args:
        direction (np.ndarray): 3D vector.
        radius (float): Radius of the cone.
        max_angle (float): Maximum angle of the cone in radians.
        ideal_spacing (float): Ideal spacing between the points.
    """

    # first we generate points along the x axis
    # we first identify rings along which to generate the points
    n_theta = int(np.round(max_angle / np.arccos(1 - (ideal_spacing / radius)**2 / 2)))
    theta = np.linspace(0, max_angle, n_theta)

    # we then generate points along the rings
    n_phi = int(np.round(2 * np.pi / np.arccos(1 - (ideal_spacing / radius)**2 / 2)))
    phi = np.linspace(0, 2 * np.pi, n_phi)

    print(n_theta, n_phi)
    # we then generate the points
    points = np.zeros((int(n_phi) * int(n_theta), 3))
    points[:, 0] = radius * np.cos(phi).repeat(n_theta) * np.sin(theta).repeat(n_phi)
    points[:, 1] = radius * np.sin(phi).repeat(n_theta) * np.sin(theta).repeat(n_phi)
    points[:, 2] = np.tile(radius * np.cos(theta), n_phi)

    return points



def calculate_angle(v1, v2, v3):
    """
    Calculate the angle at v2 formed by v1-v2-v3 in degrees.
    """
    a = np.array(v1) - np.array(v2)
    b = np.array(v3) - np.array(v2)
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # Clamp the cosine_angle to the valid range to avoid numerical issues
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle