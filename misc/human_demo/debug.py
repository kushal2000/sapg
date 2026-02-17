import pybullet as pb

def reset_base_visual_pose(body_id, target_pos, target_orn=(0, 0, 0, 1)):
    """
    Moves the body so that its visual/link origin is at target_pos/target_orn, 
    compensating for the inertial offset.
    """
    # 1. Get the offset of the CoM relative to the URDF link origin
    # dynamics_info[3] is localInertialFramePosition
    # dynamics_info[4] is localInertialFrameOrientation
    dynamics_info = pb.getDynamicsInfo(body_id, -1)
    inertial_pos = dynamics_info[3]
    inertial_orn = dynamics_info[4]
    
    # 2. Calculate the specific World CoM pose that results in the desired visual pose
    # Formula: Target_CoM = Target_Visual * Local_Inertial_Offset
    final_pos, final_orn = pb.multiplyTransforms(
        target_pos, target_orn,  # Where we want the visual origin to be
        inertial_pos, inertial_orn # The offset from visual to CoM
    )
    
    # 3. Apply the reset
    pb.resetBasePositionAndOrientation(body_id, final_pos, final_orn)

pb.connect(pb.GUI)

# V1
# robot_pb = pb.loadURDF("assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf", basePosition=(0, 0.8, 0), baseOrientation=(0, 0, 0, 1))

# V2
# robot_pb = pb.loadURDF("assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf")
# pb.resetBasePositionAndOrientation(robot_pb, (0, 0.8, 0), (0, 0, 0, 1))

# V3
robot_pb = pb.loadURDF("assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf")
reset_base_visual_pose(robot_pb, (0, 0.8, 0), (0, 0, 0, 1))

breakpoint()