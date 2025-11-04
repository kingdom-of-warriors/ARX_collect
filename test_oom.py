import torch
import traceback
import ipdb
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

model_path = "/home/arx/lerobot/single_arm_picking_pi0_sft"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
H, W = 224, 224  # 假设图像大小为 224x224


print(f"正在从 {model_path} 加载模型到 {device}...")
model = PI0Policy.from_pretrained(model_path).to(device)
model.eval()  # 切换到评估模式
print("模型加载成功。")

print("正在创建随机输入...")
input_data = {
    "observation.images.head": torch.rand(1, 3, H, W),
    "observation.images.left_wrist": torch.rand(1, 3, H, W),
    "observation.images.right_wrist": torch.rand(1, 3, H, W),
    
    "observation.state": torch.rand(1, 14),
    "observation.eef": torch.rand(1, 14),
    
    # 文本输入
    "task": ["pick up the red block"],
    "repo_id": ["dual_arm_picking"]  # 这个 ID 必须与模型归一化缓冲区中的 ID 匹配
}

input_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}

print("\n--- 正在使用修正后的输入进行推理 ---")
print("输入字典键 (Input keys):", input_data.keys())

start_mem = 0
peak_mem = 0

# 确保 CUDA 可用，并且所有之前的操作都已完成
if device.startswith("cuda"):
    torch.cuda.synchronize(device)
    
    # 1. 记录推理前的显存占用 (模型权重 + 输入张量)
    start_mem = torch.cuda.memory_allocated(device)
    
    # 2. 重置峰值统计
    torch.cuda.reset_peak_memory_stats(device)
    print(f"\n--- 准备推理 ---")
    print(f"推理前显存 (模型+输入): {start_mem / 1024**2:.2f} MB")

try:
    with torch.no_grad():  # 推理时关闭梯度计算
        output = model.select_action(input_data)
    
    print("\n--- 推理成功! ---")
    print("\n模型输出:")
    print(output)
    print("\n输出维度:")
    print(output.shape)

except Exception as e:
    print(f"\n--- 推理失败 ---")
    print(f"模型运行时出错: {e}")
    traceback.print_exc()

ipdb.set_trace()