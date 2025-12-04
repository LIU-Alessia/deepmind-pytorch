# dataset_reader.py (Fixed Alignment)
import torch
from torch.utils.data import Dataset
import numpy as np
import os

class RatMotionModel:
    def __init__(self, dt=0.02, env_size=2.2):
        self.dt = dt
        self.env_size = env_size
        self.limit = env_size / 2.0
        
        self.stdev_v = 0.13
        self.stdev_phi = np.deg2rad(330)
        self.wall_dist = 0.03
        self.slow_factor = 0.25
        self.wall_turn = np.deg2rad(90)
        
    def generate_trajectory(self, seq_len):
        # 1. 初始化 P0
        pos = np.random.uniform(-self.limit, self.limit, size=(2,))
        hd = np.random.uniform(-np.pi, np.pi) 
        
        speed_samples = np.random.rayleigh(scale=self.stdev_v, size=seq_len)
        ang_vel_samples = np.random.normal(0, self.stdev_phi, size=seq_len)
        
        # === 关键修改：先记录 P0 ===
        # pos_seq[0] 将是 P0, pos_seq[1] 是 P1...
        pos_seq = [pos.copy()]
        hd_seq = [hd]
        vel_seq = [] 
        
        curr_pos = pos.copy()
        curr_hd = hd
        
        for t in range(seq_len):
            v = speed_samples[t]
            av = ang_vel_samples[t]
            
            # Wall interaction
            dist = np.array([curr_pos[0] - (-self.limit), self.limit - curr_pos[0],
                             curr_pos[1] - (-self.limit), self.limit - curr_pos[1]])
            if np.min(dist) < self.wall_dist:
                v *= self.slow_factor
                wall_idx = np.argmin(dist)
                wall_normals = [[1,0], [-1,0], [0,1], [0,-1]]
                normal = np.array(wall_normals[wall_idx])
                heading = np.array([np.cos(curr_hd), np.sin(curr_hd)])
                if np.dot(heading, normal) < 0:
                    av += (self.wall_turn / self.dt) * np.sign(np.random.randn())
            
            # Integration
            dx = v * np.cos(curr_hd) * self.dt
            dy = v * np.sin(curr_hd) * self.dt
            curr_pos = np.clip(curr_pos + [dx, dy], -self.limit+0.001, self.limit-0.001)
            curr_hd = (curr_hd + av * self.dt + np.pi) % (2*np.pi) - np.pi
            
            # 记录 P_{t+1}
            pos_seq.append(curr_pos.copy())
            hd_seq.append(curr_hd)
            vel_seq.append([v, np.sin(av), np.cos(av)])
            
        # === 数据切分 ===
        # Init: P0 (seq的第0个)
        # Target: P1...PT (seq的第1到最后)
        # Velocity: v0...vT-1 (对应从P0->P1 到 PT-1->PT)
        
        pos_seq = np.array(pos_seq)
        hd_seq = np.array(hd_seq)
        
        return {
            'init_pos': pos_seq[0],          # P0
            'init_hd': np.array([hd_seq[0]]), # HD0
            'target_pos': pos_seq[1:],       # P1 ... PT
            'target_hd': hd_seq[1:][:, np.newaxis], # HD1 ... HDT
            'ego_vel': np.array(vel_seq)     # v0 ... vT-1
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
                    'init_pos': torch.from_numpy(traj['init_pos']).float(),
                    'init_hd': torch.from_numpy(traj['init_hd']).float(),
                    'ego_vel': torch.from_numpy(traj['ego_vel']).float(),
                    'target_pos': torch.from_numpy(traj['target_pos']).float(),
                    'target_hd': torch.from_numpy(traj['target_hd']).float()
                })
            print("")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.data, save_path)
            
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]