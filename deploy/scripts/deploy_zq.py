# SPDX-License-Identifier: BSD-3-Clause
#
# qinjian zq tech


import math
import numpy as np
import pandas as pd
from collections import deque
import torch
import time
import lcm
from pynput.keyboard import Key, Listener, Controller
from threading import Thread
from datetime import datetime

from deploy.lcm_types.pd_targets_lcmt import pd_targets_lcmt
from deploy.utils.state_estimator import StateEstimator
from deploy.utils.act_gen import ActionGenerator
from deploy.utils.ankle_joint_converter import decouple, forward_kinematics
from deploy.utils.logger import SimpleLogger

s_stepTest = False
s_stepNet = False
s_stepCalibrate = False
s_timestep = 0
leg = ActionGenerator()


def start_listener():
    thread = Thread(target=listen_keyboard)
    thread.start()


def listen_keyboard():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def on_press(key):
    global s_stepCalibrate, s_stepTest, s_stepNet, leg
    # print(f'key {key} is pressed')
    if str(key) == "'1'":
        leg.episode_length_buf[0] = 0
        s_stepCalibrate = True
        # s_stepCalibrate = False
        s_stepTest = False
        s_stepNet = False
        print('!!!!!  静态归零模式 ！')
    elif str(key) == "'2'":
        leg.episode_length_buf[0] = 0
        s_stepTest = True
        s_stepCalibrate = False
        # s_stepTest = False
        s_stepNet = False
        print('!!!!!  挂起动腿模式 ！')
    elif str(key) == "'3'":
        leg.episode_length_buf[0] = 0
        s_stepNet = True
        s_stepCalibrate = False
        s_stepTest = False
        # s_stepNet = False
        print('!!!!!  神经网络模式 ！')


def on_release(key):
    pass


# def stop_listener():
#     listener.stop()


class Deploy:
    def __init__(self, cfg, path):
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self.dof_index = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
        self.cfg = cfg
        self.df = pd.DataFrame()
        self.log_path = path

    def publish_action(self, action):
        command_for_robot = pd_targets_lcmt()
        # command_for_robot.q_des = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.double)  #action[index]
        # command_for_robot.qd_des = np.array([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1], dtype=np.double)  #np.zeros(12)
        # command_for_robot.kp = np.array([0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2], dtype=np.double)  #cfg.robot_config.kps[index]
        # command_for_robot.kd = np.array([0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3, 11.3], dtype=np.double)  #cfg.robot_config.kds[index]
        command_for_robot.q_des = action[self.dof_index]
        command_for_robot.qd_des = np.zeros(12)
        command_for_robot.kp = self.cfg.robot_config.kps[self.dof_index]
        command_for_robot.kd = self.cfg.robot_config.kds[self.dof_index]

        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0

        # 由lcm将神经网络输出的action传入c++ sdk
        self.lc.publish("robot_command", command_for_robot.encode())

    def quaternion_to_euler_array(self, quat):
        # Ensure quaternion is in the correct format [x, y, z, w]
        x, y, z, w = quat

        # Roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        # Pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        # Yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        # Returns roll, pitch, yaw in a NumPy array in radians
        return np.array([roll_x, pitch_y, yaw_z])

    def get_obs(self, es):
        """
        Extracts an observation from the mujoco data structure
        """
        q = es.joint_pos[self.dof_index].astype(np.double)
        dq = es.joint_vel[self.dof_index].astype(np.double)
        quat = es.quat[[1, 2, 3, 0]].astype(np.double)
        # r = R.from_quat(quat)
        # v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
        omega = es.omegaBody[[0, 1, 2]].astype(np.double)
        # gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        return q, dq, quat, omega

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """
        Calculates torques from position commands
        """

        return (target_q - q) * kp + (target_dq - dq) * kd

    def run_robot(self, policy):
        """
        Run the Mujoco simulation using the provided policy and configuration.

        Args:
            policy: The policy used for controlling the simulation.
            cfg: The configuration object containing simulation settings.

        Returns:
            None
        """
        # model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
        # model.opt.timestep = cfg.sim_config.dt
        # data = mujoco.MjData(model)
        # mujoco.mj_resetData(model, data)
        #
        # mujoco.mj_step(model, data)
        # viewer = mujoco_viewer.MujocoViewer(model, data)

        global s_stepCalibrate, s_stepTest, s_stepNet, s_timestep, leg
        target_q = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        action = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        action_p = np.zeros(self.cfg.env.num_actions, dtype=np.double)

        hist_obs = deque()
        for _ in range(self.cfg.env.frame_stack):
            hist_obs.append(np.zeros([1, self.cfg.env.num_single_obs], dtype=np.double))

        tau = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        current_time = time.time()

        # start thread receiving robot state
        es = StateEstimator(self.lc)
        es.spin()

        start_listener()
        sp_logger = SimpleLogger('/home/qin/Desktop/logs/deploy_logs')
        try:
            while True:
                time.sleep(max(self.cfg.env.dt - (time.time() - current_time), 0))
                if s_timestep % 100 == 0:
                    print(f'frq: {1 / (time.time() - current_time)} Hz count={s_timestep}')
                current_time = time.time()

                # Obtain an observation
                q, dq, quat, omega = self.get_obs(es)
                q = q[-self.cfg.env.num_actions:]
                # q = target_q[-12:]
                dq = dq[-self.cfg.env.num_actions:]

                obs = np.zeros([1, self.cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = self.quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi

                obs[0, 0] = math.sin(2 * math.pi * s_timestep * self.cfg.env.dt / self.cfg.env.cycle_time)
                obs[0, 1] = math.cos(2 * math.pi * s_timestep * self.cfg.env.dt / self.cfg.env.cycle_time)
                obs[0, 2] = self.cfg.cmd.vx * self.cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = self.cfg.cmd.vy * self.cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = self.cfg.cmd.dyaw * self.cfg.normalization.obs_scales.ang_vel
                obs[0, 5:17] = q * self.cfg.normalization.obs_scales.dof_pos
                obs[0, 17:29] = dq * self.cfg.normalization.obs_scales.dof_vel
                obs[0, 29:41] = action
                obs[0, 41:44] = omega
                obs[0, 44:47] = eu_ang

                obs = np.clip(obs, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)

                # 将obs写入文件，在桌面
                sp_logger.save(obs, s_timestep)

                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, self.cfg.env.num_observations], dtype=np.float32)
                for i in range(self.cfg.env.frame_stack):
                    policy_input[0, i * self.cfg.env.num_single_obs : (i + 1) * self.cfg.env.num_single_obs] = hist_obs[i][0, :]

                action_p = np.copy(q * 4)

                if s_stepCalibrate:
                    # 当状态是“校准姿态”时：将所有电机缓慢置于初始位置。12312312311
                    action = leg.calibrate(action_p)
                else:
                    if s_stepTest:
                        # 当状态是“测试姿态”时：使用动作发生器，生成腿部动作
                        action = np.array(leg.step()[0])
                    elif s_stepNet:
                        # 当状态是“行走姿态”时：使用神经网络输出动作
                        action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                    else:
                        action = np.zeros(self.cfg.env.num_actions, dtype=np.double)

                action = np.clip(action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)

                # action[4] = 0.8
                # action[10] = 0.8

                target_q = action * self.cfg.env.action_scale

                # 将神经网络生成的，左右脚的pitch、row位置，映射成关节电机角度
                my_joint_left, _ = decouple(-target_q[4], target_q[5], "left")
                my_joint_right, _ = decouple(target_q[10], target_q[11], "right")

                target_q[4] = my_joint_left[0]
                target_q[5] = my_joint_left[1]
                target_q[10] = my_joint_right[0]
                target_q[11] = my_joint_right[1]

                # target_dq = np.zeros(self.cfg.env.num_actions, dtype=np.double)
                # Generate PD control
                # tau = self.pd_control(target_q, q, self.cfg.robot_config.kps,
                #                       target_dq, dq, self.cfg.robot_config.kds)  # Calc torques
                # tau = np.clip(tau, -self.cfg.robot_config.tau_limit, self.cfg.robot_config.tau_limit)  # Clamp torques

                # !!!!!!!! send target_q to lcm
                self.publish_action(target_q)
                s_timestep += 1

        except KeyboardInterrupt:
            print(f'count={s_timestep}')
            es.close()
        finally:
            sp_logger.close()


class DeployCfg:

    class env:
        dt = 0.01
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        cycle_time = 0.64
        num_actions = 12
        action_scale = 0.25

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0

        clip_observations = 4.
        clip_actions = 4.

    class cmd:
        vx = 0.0  # 0.5
        vy = 0.0  # 0.
        dyaw = 0.0  # 0.05

    class robot_config:
        # kps = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.double)
        # kds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
        # tau_limit = 200. * np.ones(12, dtype=np.double)
        kps = np.array([200, 200, 200, 200, 50, 50, 200, 200, 200, 200, 50, 50], dtype=np.double)
        # kds = np.array([5, 5, 5, 5, 1, 1, 5, 5, 5, 5, 1, 1], dtype=np.double)
        kds = np.array([10, 10, 10, 10, 0, 0, 10, 10, 10, 10, 0, 0], dtype=np.double)

        tau_limit = 200. * np.ones(12, dtype=np.double)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    policy = torch.jit.load(args.load_model)
    deploy = Deploy(DeployCfg(), args.load_model)
    deploy.run_robot(policy)
