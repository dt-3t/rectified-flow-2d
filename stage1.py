# Rectified Flow 2d, stage 1

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# --- 1. 速度预测网络 ---
# 输入当前状态与t，输出速度
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x, t):
        # x: [bs, d]
        # t: [bs,]
        x = torch.cat([x, t.unsqueeze(1)], dim=1)  # [bs, d+1]
        pred = self.net(x)  # [bs, d]
        return pred


# --- 2. 数据准备 ---
# 生成num_points个椭圆上的点作为目标分布的数据，即训练集
def get_target_distribution(num_points):
    theta = torch.linspace(0, 2 * np.pi, num_points)
    target_x = 2 + torch.cos(theta)
    target_y = 1 + 2 * torch.sin(theta)
    return torch.stack([target_x, target_y], dim=1)


# --- 3. 训练函数 ---
def train_rectified_flow(model, target_data, num_epochs=5000, batch_size=256, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    initial_noise_dim = target_data.shape[1]

    for epoch in range(num_epochs):
        # 从初始噪声分布中采样
        z0 = torch.randn(batch_size, initial_noise_dim)  # [bs, d]

        # 随机采样时间 t，范围在 [0, 1] 之间
        t = torch.rand(batch_size, 1)  # [bs, 1]

        # 从目标分布中随机选择与当前batch_size数量相等的点作为对应的目标
        indices = torch.randint(0, target_data.shape[0], (batch_size,))  # [bs,]
        z1 = target_data[indices]  # [bs, d]

        # 使用t给目标点加噪
        xt = z0 + t * (z1 - z0)  # [bs, d]

        # 期望网络预测的是速度，即噪声到目标的差
        vt_true = z1 - z0  # [bs, d]

        # 预测速度
        vt_pred = model(xt.detach(), t.squeeze(1).detach())  # [bs, d]

        # 计算损失：预测速度与真实速度之间的均方误差
        loss = loss_fn(vt_pred, vt_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    print("Training finished.")


# --- 4. 生成函数 (多步采样) ---
def generate_samples(model, initial_noise_for_plot, num_steps=1):
    model.eval()
    num_samples = initial_noise_for_plot.shape[0]
    with torch.no_grad():
        current_samples = initial_noise_for_plot.clone()  # [m, d]
        dt = 1.0 / num_steps  # 每一步走的时间长度

        # 从 t=0 逐步模拟到 t=1
        for i in range(num_steps):
            # 当前时间点
            t = torch.full((num_samples,), i * dt, device=current_samples.device)  # [m,]

            # 预测速度
            predicted_velocity = model(current_samples, t)  # [m, d]

            # Euler方法：x_new = x_old + v * dt
            current_samples = current_samples + predicted_velocity * dt  # [m, d]

        return current_samples  # [m, d]


# --- 5. 可视化函数 ---
def plot_results(initial_samples, generated_samples, target_samples, step_title=""):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(initial_samples[:, 0], initial_samples[:, 1], alpha=0.5, label='Initial Noise (Z0)')
    plt.scatter(target_samples[:, 0], target_samples[:, 1], color='red', marker='x', label='Target Circle Points')
    plt.title('Initial Noise Distribution vs. Target')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    plt.subplot(1, 2, 2)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, label='Generated Samples')
    plt.scatter(target_samples[:, 0], target_samples[:, 1], color='red', marker='x', label='Target Circle Points')
    plt.title(f'Generated Samples vs. Target ({step_title})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.tight_layout()
    plt.show()


# --- 6. 主函数 ---
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    target_data = get_target_distribution(num_points=128)  # [n, d]，n为训练集点的数量，d为2表示是2维
    rf_model = MLP()

    print("Starting training...")
    train_rectified_flow(rf_model, target_data, num_epochs=4000, batch_size=512, lr=5e-4)

    # 推理
    num_generated_samples = 512
    initial_noise_for_plot = torch.randn(num_generated_samples, 2)  # [m, d]

    # 推理时的步数列表
    step_list = [1, 2, 4, 8, 16, 32]

    for num_steps in step_list:
        print(f"\nGenerating samples with {num_steps} steps...")
        generated_samples = generate_samples(rf_model, initial_noise_for_plot, num_steps=num_steps)
        plot_results(initial_noise_for_plot, generated_samples, target_data, step_title=f"{num_steps} Steps")

    print("\nAll simulations complete.")