import torch
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

model_path = "/home/haoming/ARX/single_arm_picking/"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
H, W = 224, 224  # 假设图像大小为 224x224

# img_tensor_test = torch.randint(0, 256, (1, 3, H, W)).float()
img_tensor = torch.rand(1, 3, H, W)

print(f"正在从 {model_path} 加载模型到 {device}...")
model = PI0Policy.from_pretrained(model_path).to(device)
model.eval()  # 切换到评估模式
print("模型加载成功。")

print("正在创建随机输入...")
input_data = {
    "observation.images.head": img_tensor,
    "observation.images.left_wrist": img_tensor,
    "observation.images.right_wrist": img_tensor,
    
    "observation.state": torch.rand(1, 14),
    "observation.eef": torch.rand(1, 14),
    
    # 文本输入
    "task": ["pick up the red block"],
    "repo_id": ["dual_arm_picking"]  # 这个 ID 必须与模型归一化缓冲区中的 ID 匹配
}

input_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}

print("\n--- 正在使用修正后的输入进行推理 ---")
print("输入字典键 (Input keys):", input_data.keys())

with torch.no_grad():  # 推理时关闭梯度计算
    output = model.select_action(input_data)

print("\n--- 推理成功! ---")
print("\n模型输出:")
print(output)
print("\n输出维度:")
import ipdb; ipdb.set_trace()
print(output.shape)