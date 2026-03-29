import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

# Initialize nuScenes through parameter instead


def get_bev_input(nusc, sample_token, grid_res=0.25, grid_size=100):
    """
    Generates a (200, 200, 3) tensor.
    Channel 0: Max Height, Channel 1: Intensity, Channel 2: Density
    """
    sample = nusc.get('sample', sample_token)
    
    # Aggregate 10 sweeps for density and motion "trails"
    lidar_points, _ = LidarPointCloud.from_file_multisweep(
        nusc, sample, chan='LIDAR_TOP', ref_chan='LIDAR_TOP', nsweeps=10
    )
    points = lidar_points.points.T # Shape (N, 4) -> [x, y, z, intensity]
    
    # Define grid boundaries (centered on car)
    limit = grid_size // 2
    mask = (points[:, 0] > -limit) & (points[:, 0] < limit) & \
           (points[:, 1] > -limit) & (points[:, 1] < limit)
    points = points[mask]
    
    pixel_count = int(grid_size / grid_res)
    bev_map = np.zeros((pixel_count, pixel_count, 4), dtype=np.float32)

    heights_dict = {}
    
    
    # Convert meters to pixel indices
    x_img = ((-points[:, 1] / grid_res) + (pixel_count / 2)).astype(np.int32)
    y_img = ((-points[:, 0] / grid_res) + (pixel_count / 2)).astype(np.int32)
    
    # initialize maximum height constant to gauge what the model can "see"
    max_h = 3.0
    min_h = -2.0
    # Fill the channels
    for i in range(len(points)):
        px, py = x_img[i], y_img[i]
        if 0 <= px < pixel_count and 0 <= py < pixel_count:
            # Channel 0: Height (Normalized roughly by a max height of 3m)
             #bev_map[py, px, 0] = min(points[i, 2]/max_h, 1.0) 
            normalized_z = np.clip((points[i, 2]- min_h) / (max_h - min_h), 0, 1)
            if points[i, 2] > bev_map[py, px, 0]:
               
                bev_map[py, px, 0] = normalized_z
            # Channel 1: Intensity
            bev_map[py, px, 1] = points[i, 3] / 255.0
            # Channel 2: Density (normalized by expected max points)
            bev_map[py, px, 2] += 0.05 
            idx = (py,px)
            

            if idx not in heights_dict: heights_dict[idx] = []
            heights_dict[idx].append(points[i, 2])

    
    # Finalize Roughness: High value = "Spiky" area, Low value = "Flat" area
    for (py, px), h_list in heights_dict.items():
        if len(h_list) > 1:
            # Use Standard Deviation as a proxy for roughness
            roughness = np.std(h_list)
            bev_map[py, px, 3] = np.clip(roughness, 0, 1)
            
    return bev_map