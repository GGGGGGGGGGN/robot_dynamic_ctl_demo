import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

class BenchmarkPlotter:
    def __init__(self, run_name="benchmark"):
        self.run_name = run_name

    def plot(self, history, save_path=None):
        """
        绘制所有关节的 Benchmark 结果，并输出数值误差统计
        """
        if save_path is None:
            save_path = f"{self.run_name}_result.png"

        # 1. 数据准备
        t = np.array(history["t"])
        q = np.array(history["q"])      
        dq = np.array(history["dq"])
        tau = np.array(history["tau"])
        q_ref = np.array(history["q_ref"])
        dq_ref = np.array(history["dq_ref"])
        
        # 计算误差 (单位: 度)
        error_deg = (q_ref - q) * 180 / np.pi
        num_joints = q.shape[1]

        # ==========================================
        # 🔥 新增：数值误差统计输出
        # ==========================================
        print(f"\n📊 --- {self.run_name} 跟踪误差统计 (单位: 度) ---")
        print(f"{'关节':<8} | {'平均误差(RMSE)':<15} | {'最大误差(Max)':<15}")
        print("-" * 50)
        
        all_rmse = []
        for i in range(num_joints):
            rmse = np.sqrt(np.mean(error_deg[:, i]**2))
            max_err = np.max(np.abs(error_deg[:, i]))
            all_rmse.append(rmse)
            print(f"Joint {i+1:<2} | {rmse:<15.4f} | {max_err:<15.4f}")
        
        print("-" * 50)
        print(f"综合平均误差 (Total RMSE): {np.mean(all_rmse):.4f} 度\n")
        # ==========================================

        # 2. 创建画布: 7 行 4 列 
        fig, axes = plt.subplots(num_joints, 4, figsize=(20, 3 * num_joints), sharex=True)
        fig.suptitle(f"Benchmark Results: {self.run_name}", fontsize=18, y=0.99)

        # 3. 循环绘制每个关节
        for i in range(num_joints):
            # --- Column 1: Position ---
            ax = axes[i, 0]
            ax.plot(t, q_ref[:, i], 'r--', label="Target", lw=1.5)
            ax.plot(t, q[:, i], 'b-', label="Real", lw=1.0)
            ax.set_ylabel(f"J{i+1} Pos (rad)")
            ax.grid(True, alpha=0.3)
            if i == 0: 
                ax.set_title("Position Tracking", fontsize=12, fontweight='bold')
                ax.legend(fontsize='x-small')

            # --- Column 2: Velocity ---
            ax = axes[i, 1]
            ax.plot(t, dq_ref[:, i], 'r--', lw=1.5)
            ax.plot(t, dq[:, i], 'b-', lw=1.0)
            ax.set_ylabel(f"J{i+1} Vel (rad/s)")
            ax.grid(True, alpha=0.3)
            if i == 0: ax.set_title("Velocity Tracking", fontsize=12, fontweight='bold')

            # --- Column 3: Torque ---
            ax = axes[i, 2]
            ax.plot(t, tau[:, i], 'g-', lw=1.0)
            ax.set_ylabel(f"J{i+1} Torque (Nm)")
            ax.grid(True, alpha=0.3)
            if i == 0: ax.set_title("Control Torque", fontsize=12, fontweight='bold')

            # --- Column 4: Error (deg) ---
            ax = axes[i, 3]
            ax.plot(t, error_deg[:, i], 'k-', lw=1.0)
            ax.set_ylabel(f"J{i+1} Err (deg)")
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='r', linestyle=':', alpha=0.5)
            if i == 0: ax.set_title("Tracking Error", fontsize=12, fontweight='bold')

        # 4. 设置底部的 X 轴标签
        for j in range(4):
            axes[-1, j].set_xlabel("Time (s)")

        plt.tight_layout()
        
        # 5. 保存
        print(f"💾 正在保存全关节结果图表到: {save_path} ...")
        plt.savefig(save_path, dpi=200)
        print(f"✅ 保存成功！")
        plt.close(fig)
        
        
        

def plot_tracking_comparison(t, ref_q, ref_dq, 
                             q1, dq1, err1, label1, color1,
                             q2, dq2, err2, label2, color2,
                             joint_idx=1, title_suffix="", save_path="comparison.png"):
    """
    专门用于对比两个控制器（如 PD 和 CTC）跟踪性能的画图工具
    """
    print("📊 正在绘制轨迹分析图...")
    plt.figure(figsize=(12, 10))

    # 子图 1：位置跟踪
    plt.subplot(3, 1, 1)
    plt.plot(t, ref_q, 'k--', linewidth=2, label='Target Position')
    plt.plot(t, q1, color1, alpha=0.7, label=label1)
    plt.plot(t, q2, color2, alpha=0.7, label=label2)
    plt.ylabel('Position (rad)', fontsize=12)
    plt.title(f'Joint {joint_idx} Position Tracking {title_suffix}', fontsize=14)
    plt.legend()
    plt.grid(True)

    # 子图 2：速度跟踪
    plt.subplot(3, 1, 2)
    plt.plot(t, ref_dq, 'k--', linewidth=2, label='Target Velocity')
    plt.plot(t, dq1, color1, alpha=0.7, label=label1)
    plt.plot(t, dq2, color2, alpha=0.7, label=label2)
    plt.ylabel('Velocity (rad/s)', fontsize=12)
    plt.title(f'Joint {joint_idx} Velocity Tracking', fontsize=14)
    plt.legend()
    plt.grid(True)

    # 子图 3：跟踪误差
    plt.subplot(3, 1, 3)
    plt.plot(t, err1, color1, alpha=0.7, label=f'{label1} Error')
    plt.plot(t, err2, color2, alpha=0.7, label=f'{label2} Error')
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Error (rad)', fontsize=12)
    plt.title(f'Joint {joint_idx} Tracking Error', fontsize=14)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
            # 5. 保存
    print(f"💾 正在保存全关节结果图表到: {save_path} ...")
    plt.savefig(save_path, dpi=200)
    print(f"✅ 保存成功！")
    