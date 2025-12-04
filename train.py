# train.py (Fully Aligned Flags Version)
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

# Local imports
from ensembles import PlaceCellEnsemble, HeadDirectionCellEnsemble
from model import GridCellsRNN
from dataset_reader import SyntheticGridCellsDataset, RatMotionModel
import scores

# 尝试导入绘图工具
try:
    from utils import get_scores_and_plot, plot_trajectory_comparison
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# === 1. 参数定义 (完全对齐 DeepMind tf.flags) ===
parser = argparse.ArgumentParser(description='Grid Cells Supervised Training')

# Task config
parser.add_argument('--task_dataset_info', type=str, default='square_room',
                    help='Name of the room in which the experiment is performed.')
parser.add_argument('--task_root', type=str, default=None,
                    help='Dataset path (mapped to save_path logic in this repo).')
parser.add_argument('--task_env_size', type=float, default=2.2,
                    help='Environment size (meters).')
parser.add_argument('--task_n_pc', type=int, nargs='+', default=[256],
                    help='Number of target place cells (list).')
parser.add_argument('--task_pc_scale', type=float, nargs='+', default=[0.01],
                    help='Place cell standard deviation parameter (meters) (list).')
parser.add_argument('--task_n_hdc', type=int, nargs='+', default=[12],
                    help='Number of target head direction cells (list).')
parser.add_argument('--task_hdc_concentration', type=float, nargs='+', default=[20.],
                    help='Head direction concentration parameter (list).')
parser.add_argument('--task_neurons_seed', type=int, default=8341,
                    help='Seeds.')
parser.add_argument('--task_targets_type', type=str, default='softmax',
                    choices=['softmax', 'voronoi', 'sample', 'normalized'],
                    help='Type of target, soft or hard.')
parser.add_argument('--task_lstm_init_type', type=str, default='softmax',
                    choices=['softmax', 'voronoi', 'sample', 'normalized', 'zeros'],
                    help='Type of LSTM initialisation, soft or hard.')
parser.add_argument('--task_velocity_inputs', action='store_true', default=True,
                    help='Input velocity.')
parser.add_argument('--task_velocity_noise', type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help='Add noise to velocity [v, sin, cos].')

# Model config
parser.add_argument('--model_nh_lstm', type=int, default=128,
                    help='Number of hidden units in LSTM.')
parser.add_argument('--model_nh_bottleneck', type=int, default=512,
                    help='Number of hidden units in linear bottleneck.')
parser.add_argument('--model_dropout_rates', type=float, nargs='+', default=[0.5],
                    help='List of floats with dropout rates.')
parser.add_argument('--model_weight_decay', type=float, default=1e-5,
                    help='Weight decay regularisation')
parser.add_argument('--model_bottleneck_has_bias', action='store_true', default=False,
                    help='Whether to include a bias in linear bottleneck')
parser.add_argument('--model_init_weight_disp', type=float, default=0.0,
                    help='Initial weight displacement.')

# Training config
# 原版1000 epochs (1M steps)
parser.add_argument('--training_epochs', type=int, default=1000, 
                    help='Number of training epochs.') 
parser.add_argument('--training_steps_per_epoch', type=int, default=1000,
                    help='Number of optimization steps per epoch.')
parser.add_argument('--training_minibatch_size', type=int, default=10,
                    help='Size of the training minibatch.')
parser.add_argument('--training_evaluation_minibatch_size', type=int, default=4000,
                    help='Size of the minibatch during evaluation.')
parser.add_argument('--training_clipping', type=float, default=1e-5,
                    help='The absolute value to clip by.')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='Learning rate.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Momentum parameter.')

# Store
parser.add_argument('--saver_results_directory', type=str, default='./results',
                    help='Path to directory for saving results.')
parser.add_argument('--saver_eval_time', type=int, default=20,
                    help='Frequency at which results are saved.')

# Extra (Helper for this implementation)
parser.add_argument('--data_path', type=str, default='./data/grid_cells_data.pt',
                    help='Local path to save/load synthesized data.')

FLAGS = parser.parse_args()

def save_checkpoint(model, optimizer, epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, 'checkpoint_latest.pth')
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'flags': vars(FLAGS)
    }
    torch.save(state, path)

def decode_position(logits, cell_centers):
    """Softmax Center of Mass Decoding"""
    probs = torch.softmax(logits, dim=-1)
    centers = cell_centers.unsqueeze(0).unsqueeze(0) 
    pred_pos = torch.sum(probs.unsqueeze(-1) * centers, dim=2)
    return pred_pos

def visualize_figure1b(model, device, save_dir, epoch, env_size):
    """复现 Fig 1b"""
    model.eval()
    motion_model = RatMotionModel(dt=0.02, env_size=env_size)
    traj = motion_model.generate_trajectory(750) 
    
    ego_vel = torch.from_numpy(traj['ego_vel']).float().unsqueeze(0).to(device)
    init_pos = torch.from_numpy(traj['target_pos'][0]).float().unsqueeze(0).to(device)
    init_hd = torch.from_numpy(traj['target_hd'][0]).float().unsqueeze(0).to(device)
    
    # 初始化逻辑：只使用第一个 PC 和 第一个 HD 集合来初始化 (通常假设只有一个)
    # 如果有多尺度，通常只用最粗或最细的初始化，或者全部？原版 utils.py encode_initial_conditions 会 concat 所有
    init_conds = []
    # 遍历所有目标 Ensemble 进行初始化
    for ens in model.target_ensembles:
        # 判断是 PC 还是 HD
        if isinstance(ens, PlaceCellEnsemble):
            init_conds.append(ens.get_init(init_pos.unsqueeze(1)).squeeze(1))
        elif isinstance(ens, HeadDirectionCellEnsemble):
            init_conds.append(ens.get_init(init_hd.unsqueeze(1)).squeeze(1))
            
    concat_init = torch.cat(init_conds, dim=1)
    
    with torch.no_grad():
        preds, _, _ = model(concat_init, ego_vel)
        # 默认取第一个 Place Cell Ensemble 的预测来解码
        pc_logits = preds[0] 
        
    # 获取第一个 PC Ensemble 的中心
    pc_ens = model.target_ensembles[0]
    decoded = decode_position(pc_logits, pc_ens.means).squeeze(0).cpu().numpy()
    truth = traj['target_pos']
    
    if UTILS_AVAILABLE:
        filename = os.path.join(save_dir, f"fig1b_epoch_{epoch}.png")
        plot_trajectory_comparison(truth, decoded, filename, epoch)

def train():
    torch.manual_seed(FLAGS.task_neurons_seed)
    np.random.seed(FLAGS.task_neurons_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    if not os.path.exists(FLAGS.saver_results_directory):
        os.makedirs(FLAGS.saver_results_directory)

    # === 2. 初始化 Ensembles (支持多尺度/列表) ===
    # 对应 utils.get_place_cell_ensembles
    place_cell_ensembles = [
        PlaceCellEnsemble(
            n, stdev=s, 
            pos_min=-FLAGS.task_env_size/2.0, pos_max=FLAGS.task_env_size/2.0,
            seed=FLAGS.task_neurons_seed,
            soft_targets=FLAGS.task_targets_type,
            soft_init=FLAGS.task_lstm_init_type,
            device=device
        )
        for n, s in zip(FLAGS.task_n_pc, FLAGS.task_pc_scale)
    ]

    # 对应 utils.get_head_direction_ensembles
    head_direction_ensembles = [
        HeadDirectionCellEnsemble(
            n, concentration=c,
            seed=FLAGS.task_neurons_seed,
            soft_targets=FLAGS.task_targets_type,
            soft_init=FLAGS.task_lstm_init_type,
            device=device
        )
        for n, c in zip(FLAGS.task_n_hdc, FLAGS.task_hdc_concentration)
    ]
    
    target_ensembles = place_cell_ensembles + head_direction_ensembles

    # === 3. 模型初始化 ===
    model = GridCellsRNN(
        target_ensembles=target_ensembles, 
        nh_lstm=FLAGS.model_nh_lstm, 
        nh_bottleneck=FLAGS.model_nh_bottleneck, 
        dropout_rates=FLAGS.model_dropout_rates,          
        bottleneck_has_bias=FLAGS.model_bottleneck_has_bias, 
        weight_decay=FLAGS.model_weight_decay,
        init_weight_disp=FLAGS.model_init_weight_disp
    )
    model.to(device)
    
    # === 4. 优化器 ===
    optimizer = optim.RMSprop(model.get_param_groups(), 
                              lr=FLAGS.learning_rate, 
                              alpha=0.9, # Match TF default decay
                              momentum=FLAGS.momentum, 
                              eps=1e-10)

    # === 5. 数据集 ===
    print(f"Preparing Dataset...", flush=True)
    dataset_size = 100000 
    dataset = SyntheticGridCellsDataset(
        size=dataset_size, 
        sequence_length=100, # Fixed trajectory length
        env_size=FLAGS.task_env_size, 
        dt=0.02, 
        save_path=FLAGS.data_path
    )
    dataloader = DataLoader(dataset, batch_size=FLAGS.training_minibatch_size, shuffle=True)
    
    # === 6. 评分器 ===
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    # 模拟 data_reader.get_coord_range()
    limit = FLAGS.task_env_size / 2.0
    coord_range = ((-limit, limit), (-limit, limit))
    scorer = scores.GridScorer(20, coord_range, masks_parameters)

    print(f"Starting training...", flush=True)
    
    data_iter = iter(dataloader)
    total_steps = 0
    vel_noise_scale = torch.tensor(FLAGS.task_velocity_noise, device=device).float()
    
    for epoch in range(FLAGS.training_epochs):
        model.train()
        loss_acc = []
        
        for _ in range(FLAGS.training_steps_per_epoch):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Data
            ego_vel = batch['ego_vel'].to(device)
            target_pos = batch['target_pos'].to(device)
            target_hd = batch['target_hd'].to(device)
            init_pos = batch['init_pos'].to(device)
            init_hd = batch['init_hd'].to(device)
            
            # Noise
            if FLAGS.task_velocity_inputs and torch.any(vel_noise_scale > 0):
                noise = torch.randn_like(ego_vel) * vel_noise_scale
                ego_vel = ego_vel + noise
            
            # Init State (Concat all ensemble inits)
            init_conds = []
            for ens in place_cell_ensembles:
                init_conds.append(ens.get_init(init_pos.unsqueeze(1)).squeeze(1))
            for ens in head_direction_ensembles:
                init_conds.append(ens.get_init(init_hd.unsqueeze(1)).squeeze(1))
            concat_init = torch.cat(init_conds, dim=1)
            
            # Forward
            preds, bottleneck, _ = model(concat_init, ego_vel)
            
            # Loss (Dynamic loop over all ensembles)
            total_loss = 0
            # preds is list: [pc1, pc2..., hd1, hd2...]
            # target_ensembles is list: [pc1, pc2..., hd1, hd2...]
            # We map input targets to ensembles. 
            # Note: SyntheticDataset only provides 1 'target_pos' and 1 'target_hd'.
            # If multiple PC ensembles exist (multi-scale), they all define prob distributions over the SAME pos.
            
            pred_idx = 0
            for ens in place_cell_ensembles:
                target = ens.get_targets(target_pos)
                total_loss += ens.loss(preds[pred_idx], target)
                pred_idx += 1
                
            for ens in head_direction_ensembles:
                target = ens.get_targets(target_hd)
                total_loss += ens.loss(preds[pred_idx], target)
                pred_idx += 1
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), FLAGS.training_clipping)
            optimizer.step()
            
            loss_acc.append(total_loss.item())
            total_steps += 1
            
        print(f"Epoch {epoch} ({total_steps} updates). Mean Loss: {np.mean(loss_acc):.5f}", flush=True)
        
        # === Evaluation ===
        if epoch % FLAGS.saver_eval_time == 0 and UTILS_AVAILABLE:
            save_checkpoint(model, optimizer, epoch, FLAGS.saver_results_directory)
            
            # Fig 1b
            visualize_figure1b(model, device, FLAGS.saver_results_directory, epoch, FLAGS.task_env_size)
            
            # Fig 1d (Rate Maps)
            model.eval()
            with torch.no_grad():
                all_pos = []
                all_act = []
                target_samples = FLAGS.training_evaluation_minibatch_size
                eval_batches = target_samples // FLAGS.training_minibatch_size
                
                print(f"Collecting {target_samples} samples for Rate Maps...", flush=True)
                eval_iter = iter(dataloader)
                for _ in range(eval_batches):
                    try:
                        batch = next(eval_iter)
                    except StopIteration:
                        eval_iter = iter(dataloader)
                        batch = next(eval_iter)
                        
                    ego_vel = batch['ego_vel'].to(device)
                    init_pos = batch['init_pos'].to(device)
                    init_hd = batch['init_hd'].to(device)
                    
                    # Init logic same as training
                    init_conds = []
                    for ens in place_cell_ensembles:
                        init_conds.append(ens.get_init(init_pos.unsqueeze(1)).squeeze(1))
                    for ens in head_direction_ensembles:
                        init_conds.append(ens.get_init(init_hd.unsqueeze(1)).squeeze(1))
                    concat_init = torch.cat(init_conds, dim=1)
                    
                    _, bottleneck, _ = model(concat_init, ego_vel)
                    
                    all_pos.append(batch['target_pos'].cpu().numpy())
                    all_act.append(bottleneck.cpu().numpy())
                
                if len(all_pos) > 0:
                    filename = f"ratemaps_epoch_{epoch}.pdf"
                    get_scores_and_plot(scorer, np.concatenate(all_pos), np.concatenate(all_act), 
                                        FLAGS.saver_results_directory, filename)

if __name__ == '__main__':
    train()