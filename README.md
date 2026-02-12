# robot_dynamic_ctl_demo
演示基于panda机械臂的几种力控算法，便于对控制算法的学习和后期开发


mamba create -n rm_ctl python=3.13
mamba activate rm_ctl

pip install -e .

从urdf文件的获取机器人惯量信息，构建了xml文件，添加关节阻尼和驱动器的设置。


