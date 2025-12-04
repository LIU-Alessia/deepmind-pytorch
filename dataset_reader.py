# dataset_reader.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class RatMotionModel:
    """
    Simulates rat motion based on parameters from Supplementary Table 1.
    """
    def __init__(self, dt=0.02, env_size=2.2):
        self.dt = dt
        self.env_size = env_size
        self.limit = env_size / 2.0
        
        # Parameters from Table 1
        self.stdev_v = 0.13                 # Forward velocity Rayleigh scale (m/s)
        self.stdev_phi = np.deg2rad(330)    # Rotation velocity Gaussian std (rad/s)
        self.wall_dist = 0.03               # Perimeter distance (m)
        self.slow_factor = 0.25             # Velocity reduction in perimeter
        self.wall_turn = np.deg2rad(90)     # Turn angle in perimeter
        
    def generate_trajectory(self, seq_len):
        # Initialization
        pos = np.random.uniform(-self.limit, self.limit, size=(2,))
        hd = np.random.uniform(-np.pi, np.pi) 
        
        # Pre-generate velocities
        speed_samples = np.random.rayleigh(scale=self.stdev_v, size=seq_len)
        ang_vel_samples = np.random.normal(0, self.stdev_phi, size=seq_len)
        
        pos_seq, hd_seq, vel_seq = [], [], []
        curr_pos, curr_hd = pos.copy(), hd
        
        for t in range(seq_len):
            v = speed_samples[t]
            av = ang_vel_samples[t]
            
            # Wall Interaction
            # Distances to [Left, Right, Bottom, Top]
            dist = np.array([
                curr_pos[0] - (-self.limit), 
                self.limit - curr_pos[0],
                curr_pos[1] - (-self.limit), 
                self.limit - curr_pos[1]
            ])
            min_dist_idx = np.argmin(dist)
            min_dist = dist[min_dist_idx]
            
            if min_dist < self.wall_dist:
                # 1. Slow down
                v *= self.slow_factor
                
                # 2. Avoid wall (Approximating the 90 deg turn)
                # Normals: Left(1,0), Right(-1,0), Bottom(0,1), Top(0,-1)
                normals = [[1,0], [-1,0], [0,1], [0,-1]]
                wall_n = np.array(normals[min_dist_idx])
                
                # Current heading vector
                heading = np.array([np.cos(curr_hd), np.sin(curr_hd)])
                
                # If moving towards wall (dot product < 0), turn
                if np.dot(heading, wall_n) < 0:
                    # Add a strong angular velocity impulse
                    # Direction is random or based on cross product (here random sign for simplicity)
                    turn_dir = np.sign(np.random.randn())
                    av += turn_dir * (self.wall_turn / self.dt) * 0.1 # Smooth factor
            
            # Update Position
            dx = v * np.cos(curr_hd) * self.dt
            dy = v * np.sin(curr_hd) * self.dt
            new_pos = curr_pos + np.array([dx, dy])
            
            # Clip to stay in box
            curr_pos = np.clip(new_pos, -self.limit+1e-4, self.limit-1e-4)
            
            # Update Angle
            curr_hd = (curr_hd + av * self.dt + np.pi) % (2*np.pi) - np.pi
            
            # Store
            pos_seq.append(curr_pos.copy())
            hd_seq.append(curr_hd)
            vel_seq.append([v, np.sin(av), np.cos(av)])
            
        return {
            'target_pos': np.array(pos_seq), 
            'target_hd': np.array(hd_seq)[:, np.newaxis],
            'ego_vel': np.array(vel_seq)
        }

class SyntheticGridCellsDataset(Dataset):
    def __init__(self, size=100000, sequence_length=100, env_size=2.2, dt=0.02, save_path="data/traj_data.pt"):
        self.data = []
        if os.path.exists(save_path):
            print(f"[Dataset] Loading {save_path}...")
            self.data = torch.load(save_path)
        else:
            print(f"[Dataset] Generating {size} trajectories...")
            motion_model = RatMotionModel(dt, env_size)
            for i in range(size):
                if i % 1000 == 0: print(f"{i}/{size}", end='\r')
                traj = motion_model.generate_trajectory(sequence_length)
                
                self.data.append({
                    'init_pos': torch.from_numpy(traj['target_pos'][0]).float(),
                    'init_hd': torch.from_numpy(traj['target_hd'][0]).float(),
                    'ego_vel': torch.from_numpy(traj['ego_vel']).float(),
                    'target_pos': torch.from_numpy(traj['target_pos']).float(),
                    'target_hd': torch.from_numpy(traj['target_hd']).float()
                })
            print("")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.data, save_path)
            
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]