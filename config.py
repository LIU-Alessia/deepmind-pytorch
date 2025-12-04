import numpy as np

class Config:
    # --- 环境与物理参数 (Supplementary Table 1) ---
    ENV_SIZE = 2.2          # 环境边长 (米)
    WALL_LIMIT = ENV_SIZE / 2.0
    DT = 0.02               # 仿真步长 (秒)
    
    # 运动模型
    RAYLEIGH_SCALE = 0.13   # 瑞利分布尺度 (m/s)
    ROT_VEL_STD = np.radians(330) # 角速度标准差 (rad/s)
    PERIMETER_DIST = 0.03   # 边界区距离 (m)
    TURN_ANGLE = np.radians(90)   # 撞墙转向角度

    # --- 训练参数 ---
    STEPS_PER_TRAJ = 100    # BPTT 展开步数 (序列长度)
    BATCH_SIZE = 10         # Batch Size
    EPOCHS = 1000           # 训练轮数 (可增加)
    STEPS_PER_EPOCH = 1000  # 每轮迭代次数
    
    # --- 模型参数 ---
    LSTM_HIDDEN_SIZE = 128
    LINEAR_SIZE = 256       # 瓶颈层 (Bottleneck) 大小
    DROPOUT_RATE = 0.5      # Dropout 是形成 Grid Cells 的关键
    WEIGHT_DECAY = 1e-5     # L2 正则化
    LEARNING_RATE = 1e-5
    MOMENTUM = 0.9
    GRAD_CLIP_VALUE = 1e-5  # 梯度裁剪阈值

    # --- 目标编码参数 ---
    N_PLACE_CELLS = 256     # 位置细胞数量
    PLACE_CELL_SCALE = 0.01 # 位置细胞高斯分布的标准差 (m)
    N_HD_CELLS = 12         # 头朝向细胞数量
    HD_CONCENTRATION = 20.0 # Von Mises 分布的浓度参数 kappa

    # --- 系统 ---
    DEVICE = "cuda"         # 强制使用 CUDA
    SEED = 8341             # 复现随机种子
    SAVE_DIR = "./results_pytorch"