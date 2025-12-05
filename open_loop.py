import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import matplotlib.pyplot as plt
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from tqdm import tqdm

# --- 配置路径和参数 ---
LEROBOT_PI0_PATH = "/home/haoming/ARX/pi0_sft/single_arm_picking_pi0_sft_v4"
DATASET_REPO_ID = "picking_blocking_v1" 
DATASET_ROOT = "/mnt/petrelfs/lijiarui/lerobot/dataset/lerobot_dataset/picking_blocking_v2_task_new" # 如果是本地数据，填root

TASK_DESCRIPTION = "Use the right arm to pick up the blue block and place it in the transparent box"
NORM_STATS_KEY = "picking_blocking_v4"

device = "cuda" if torch.cuda.is_available() else "cpu"
episode_index_to_plot = 0  # 想画第几个 episode (0, 1, 2...)

print(f"Loading model from {LEROBOT_PI0_PATH}...")
lerobot_pi0_model = PI0Policy.from_pretrained(LEROBOT_PI0_PATH)
lerobot_pi0_model.to(device)
lerobot_pi0_model.eval()

print(f"Loading dataset {DATASET_REPO_ID}...")
dataset = LeRobotDataset(repo_id=DATASET_REPO_ID, root=DATASET_ROOT)

from_idx = dataset.episode_data_index["from"][episode_index_to_plot].item()
to_idx = dataset.episode_data_index["to"][episode_index_to_plot].item()

print(f"Processing Episode {episode_index_to_plot} (Index {from_idx} to {to_idx})...")

ground_truth_actions = []
predicted_actions = []

with torch.inference_mode():
    for idx in tqdm(range(from_idx, to_idx)):
        item = dataset[idx]
        
        batch = {
            "observation.images.head": item["observation.images.head"].unsqueeze(0).to(device),
            "observation.images.left_wrist": item["observation.images.left_wrist"].unsqueeze(0).to(device),
            "observation.images.right_wrist": item["observation.images.right_wrist"].unsqueeze(0).to(device),
            
            "observation.state": item["observation.state"].unsqueeze(0).to(device),
            "observation.eef": item["observation.eef"].unsqueeze(0).to(device),
            
            # PI0 特有的文本和归一化ID输入
            "task": [TASK_DESCRIPTION], 
            "repo_id": [NORM_STATS_KEY]
        }
        pred_action = lerobot_pi0_model.select_action(batch)
        predicted_actions.append(pred_action.squeeze(0).cpu().numpy())
        ground_truth_actions.append(item["action"].numpy())

ground_truth_actions = np.array(ground_truth_actions)
predicted_actions = np.array(predicted_actions)

print(f"GT Shape: {ground_truth_actions.shape}, Pred Shape: {predicted_actions.shape}")

action_dim = ground_truth_actions.shape[1]
num_plots = min(action_dim, 14) # 最多画14个图，防止太多
fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots), sharex=True)
if num_plots == 1: axes = [axes]

time_steps = np.arange(len(ground_truth_actions))
for dim in range(num_plots):
    axes[dim].plot(time_steps, ground_truth_actions[:, dim], label="Ground Truth", color="black", linestyle="--", alpha=0.7)
    axes[dim].plot(time_steps, predicted_actions[:, dim], label="Predicted (Open Loop)", color="red", alpha=0.8)
    axes[dim].set_ylabel(f"Dim {dim}")
    axes[dim].grid(True, alpha=0.3)
    if dim == 0:
        axes[dim].legend()

plt.xlabel("Time Steps")
plt.suptitle(f"Open Loop Evaluation: Episode {episode_index_to_plot}")
plt.tight_layout()

save_path = "open_loop_eval.png"
plt.savefig(save_path)
print(f"Plot saved to {save_path}")