# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import numpy as np
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class ZqCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47  # 47
        # num_observations = int(frame_stack * num_single_obs)
        num_observations = 47
        single_num_privileged_obs = 73  # 73
        # num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_privileged_obs = 73
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False
        env_spacing = 1.
        is_delay_obs = True  # 控制上行delay的开关
        is_delay_act = False  # 控制下行delay的开关
        queue_len_obs = 6   # 不可小于2，可以通过上面的is_delay_obs控制开关
        queue_len_act = 3   # 不可小于2，可以通过上面的is_delay_act控制开关

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [-3, -3, 3]  # [m]
        lookat = [0., 0, 1.]  # [m]

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/ZQ_Humanoid/urdf/ZQ_Humanoid_long_foot.urdf'

        name = "zq01"
        foot_name = "foot"
        knee_name = "4"

        terminate_after_contacts_on = []
        penalize_contacts_on = ['3', '4']
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = True
        fix_base_link = False
        terminate_body_height = 0.4

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 1.0
        dynamic_friction = 1.0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.85]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'JOINT_Y1': -0.03,
            'JOINT_Y2': 0.0,
            'JOINT_Y3': 0.21,
            'JOINT_Y4': -0.53,
            'JOINT_Y5': 0.31,
            'JOINT_Y6': 0.03,

            'JOINT_Z1': 0.03,
            'JOINT_Z2': 0.0,
            'JOINT_Z3': 0.21,
            'JOINT_Z4': -0.53,
            'JOINT_Z5': 0.31,
            'JOINT_Z6': -0.03,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'JOINT_Y1': 200.0, 'JOINT_Y2': 200.0, 'JOINT_Y3': 200.0, 'JOINT_Y4': 200.0, 'JOINT_Y5': 200, 'JOINT_Y6': 200,
                     'JOINT_Z1': 200.0, 'JOINT_Z2': 200.0, 'JOINT_Z3': 200.0, 'JOINT_Z4': 200.0, 'JOINT_Z5': 200, 'JOINT_Z6': 200}
        damping = {'JOINT_Y1': 10, 'JOINT_Y2': 10, 'JOINT_Y3': 10, 'JOINT_Y4': 10, 'JOINT_Y5': 4, 'JOINT_Y6': 4,
                   'JOINT_Z1': 10, 'JOINT_Z2': 10, 'JOINT_Z3': 10, 'JOINT_Z4': 10, 'JOINT_Z5': 4, 'JOINT_Z6': 4}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.005  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 6
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 1

    class domain_rand:
        randomize_friction = True
        friction_range = [0.8, 1.2]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.4
        dynamic_randomization = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:

            lin_vel_x = [-0.3, 0.6]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [-3.14, 3.14]

            # lin_vel_x = [-0.0, 0.0]  # min max [m/s]
            # lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            # ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            # heading = [-0.0, 0.0]

    class rewards:
        base_height_target = 0.83
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.50    # rad
        target_feet_height = 0.15       # m
        step_freq = 1.5                # Hz, sec=0.666
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5.
        max_contact_force = 700  # forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 1.6
            feet_clearance = 5.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = -1.2
            base_acc = 0.2
            lin_vel_z = -2.0
            # energy
            # action_smoothness = -0.002
            action_rate = -0.01
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 1.
            ang_vel = 1
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class ZqCfgPPO(LeggedRobotCfgPPO):
    seed = -1
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 3001  # number of policy updates

        # logging
        save_interval = 200  # check for potential saves every this many iterations
        experiment_name = 'zq'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
