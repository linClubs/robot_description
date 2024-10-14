#!/home/lin/software/miniconda3/envs/pin/bin/python
#coding=utf-8
import casadi                                                                       
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin                             
import time
from pinocchio import casadi as cpin                
# from pinocchio.robot_wrapper import RobotWrapper    # 可视化用的
# from pinocchio.visualize import MeshcatVisualizer   
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

np.set_printoptions(precision=5, suppress=True, linewidth=200)

class Arm_IK:
    def __init__(self, urdf_dir):
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_dir)
        # self.robot = pin.RobotWrapper.BuildFromURDF(urdf_dir + '/urdf/h1_with_hand.urdf', urdf_dir)
        
        '''
            锁定urdf中不参与逆解的joint, 记住这里不是link 
            "torso_joint",躯干也需要锁住,不然逆解躯干会跟随运动
        '''
        self.mixed_jointsToLockIDs = [  
            "J_head_yaw",
            "J_head_pitch",
            # "J_arm_r_01",
            # "J_arm_r_02",
            # "J_arm_r_03",
            # "J_arm_r_04",
            # "J_arm_r_05",
            # "J_arm_r_06",
            # "J_arm_r_07",
            # "J_arm_l_01",
            # "J_arm_l_02",
            # "J_arm_l_03",
            # "J_arm_l_04",
            # "J_arm_l_05",
            # "J_arm_l_06",
            # "J_arm_l_07",
            "J_waist_pitch",
            "J_waist_roll",
            "J_waist_yaw",
            "J_hip_r_roll",
            "J_hip_r_yaw",
            "J_hip_r_pitch",
            "J_knee_r_pitch",
            "J_ankle_r_pitch",
            "J_ankle_r_roll",
            "J_hip_l_roll",
            "J_hip_l_yaw",
            "J_hip_l_pitch",
            "J_knee_l_pitch",
            "J_ankle_l_pitch",
            "J_ankle_l_roll",
                                      ]

        # 重新创建新的urdf文件
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # for i, joint in enumerate(self.reduced_robot.model.joints):
        #     joint_name = self.reduced_robot.model.names[i]
        #     print(f"Joint {i}: {joint_name}, ID: {joint.id}")

        # 创建左边的末端link, 因为h1的臂最后一个关节为left_elbow_joint, 
        # 这个关节的子link加一个末端frame, 求出来的才算末端的姿态
        # 初始旋转为单位阵，平移为link的长度
        self.reduced_robot.model.addFrame(
            pin.Frame('L_ee',
                      self.reduced_robot.model.getJointId('J_arm_l_07'),
                      pin.SE3(np.eye(3),
                              np.array([[0.03, 0, 0]]).T),
                      pin.FrameType.OP_FRAME)
        )
        
        self.reduced_robot.model.addFrame(
            pin.Frame('R_ee',
                      self.reduced_robot.model.getJointId('J_arm_r_07'),
                      pin.SE3(np.eye(3),
                              np.array([[0.03, 0, 0]]).T),
                      pin.FrameType.OP_FRAME)
        )
        # print("1111", self.reduced_robot.model.getJointId('left_elbow_joint'),  self.reduced_robot.model.getJointId('right_elbow_joint'))
        # 初始时刻的qpose设为0
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.init_data[4] = -0.5
        self.init_data[8] = -0.5

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)

        # 前向运动学
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # 获得末端执行器的id，Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        
        # 添加损失函数
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.L_hand_id].inverse() * cpin.SE3(self.cTf_l)
                    ).vector[:3],
                    cpin.log6(
                        self.cdata.oMf[self.R_hand_id].inverse() * cpin.SE3(self.cTf_r)
                    ).vector[:3]
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        # self.param_q_ik_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization = casadi.sumsqr(self.var_q)
        # self.smooth_cost = casadi.sumsqr(self.var_q - self.param_q_ik_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(10 * self.totalcost + 0.001 * self.regularization)
        # self.opti.minimize(20 * self.totalcost + 0.001*self.regularization + 0.1*self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':500,
                'tol':1e-4
            },
            'print_time':False
        }
        self.opti.solver("ipopt", opts)

    def adjust_pose(self, human_left_pose, human_right_pose, human_arm_length=0.55, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def ik_fun(self, left_pose, right_pose, motorstate=None, motorV=None):
        if motorstate is not None:
            self.init_data = motorstate
            self.init_data[4] = -0.5
            self.init_data[8] = -0.5
        self.opti.set_initial(self.var_q, self.init_data)

        # self.vis.viewer['L_ee_target'].set_transform(left_pose)   # for visualization
        # self.vis.viewer['R_ee_target'].set_transform(right_pose)  # for visualization

        # left_pose, right_pose = self.adjust_pose(left_pose, right_pose)

        self.opti.set_value(self.param_tf_l, left_pose)
        self.opti.set_value(self.param_tf_r, right_pose)

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            # self.vis.display(sol_q)  # for visualization
            self.init_data = sol_q

            if motorV is not None:
                v =motorV * 0.0
            else:
                v = (sol_q-self.init_data ) * 0.0

            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q,v,np.zeros(self.reduced_robot.model.nv))

            return sol_q, tau_ff ,True
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            return sol_q, '',False

if __name__ == "__main__":
    urdf_path = "/home/lin/ros_code/h1_ws/src/h1_teleop/robot_description/urdf/qingloong/qingloong.urdf"
    arm_ik = Arm_IK(urdf_path)

    poseL = pin.SE3(pin.Quaternion(0.456, 0.541, 0.540, -0.456), np.array([0.328, 0.226, 0.068]))
    poseR = pin.SE3(pin.Quaternion(-0.456, 0.541, -0.540, -0.456), np.array([0.328, -0.226, 0.068]))
    
    count = 0
    while True:
        z = 0.068 + 0.3 * np.sin(0.01*count)
        poseL.translation[2] = z
        poseR.translation[2] = z
        print(poseL.translation)
        print(poseL.rotation)
        qpos, _, _ = arm_ik.ik_fun(poseL.homogeneous, poseR.homogeneous)
        print(qpos)
        count += 1
