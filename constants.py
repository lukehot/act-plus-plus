import pathlib
import os

### Task parameters
DATA_DIR = "/home/luke/robot/act-plus-plus/act_data"
SIM_TASK_CONFIGS = {
    "sim_transfer_cube_scripted": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_scripted",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top", "left_wrist", "right_wrist"],
    },
    "sim_transfer_cube_top_3_camera": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_top_3_camera",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top", "top_1", "top_2"],
    },
    "sim_transfer_cube_top_left_camera": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_top_left_camera",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top", "left_wrist"],
    },
    "S1_pickup": {
        "dataset_dir": DATA_DIR + "/S1_pickup",
        "num_episodes": 1,
        "episode_len": 250,
        "camera_names": ["top", "angle"],
    },
    "sim_pickup": {
        "dataset_dir": DATA_DIR + "/sim_pickup",
        "num_episodes": 50,
        "episode_len": 200,
        "camera_names": ["top", "wrist"],
        "action_dim": 9,
        "camera_names": ["top", "wrist"],
    },
    "sim_transfer_cube_human": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_human",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top"],
    },
    "sim_insertion_scripted": {
        "dataset_dir": DATA_DIR + "/sim_insertion_scripted",
        "num_episodes": 50,
        "episode_len": 400,
        "camera_names": ["top", "left_wrist", "right_wrist"],
    },
    "sim_insertion_human": {
        "dataset_dir": DATA_DIR + "/sim_insertion_human",
        "num_episodes": 50,
        "episode_len": 500,
        "camera_names": ["top"],
    },
    "all": {
        "dataset_dir": DATA_DIR + "/",
        "num_episodes": None,
        "episode_len": None,
        "name_filter": lambda n: "sim" not in n,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    },
    "sim_transfer_cube_scripted_mirror": {
        "dataset_dir": DATA_DIR + "/sim_transfer_cube_scripted_mirror",
        "num_episodes": None,
        "episode_len": 400,
        "camera_names": ["top", "left_wrist", "right_wrist"],
    },
    "sim_insertion_scripted_mirror": {
        "dataset_dir": DATA_DIR + "/sim_insertion_scripted_mirror",
        "num_episodes": None,
        "episode_len": 400,
        "camera_names": ["top", "left_wrist", "right_wrist"],
    },
}

### Simulation envs fixed constants
DT = 0.02
FPS = 50
JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
# gripper + wheel
S1_START_QPOS = [0] * 14 + [0.021, -0.021] + [0] * 2
# S1_MOCAP_START_QPOS = [-0.030, -0.17, 0.15]
# S1_MOCAP_START_QPOS = [0.34724606, 1.2338852, 0.80517225]
#   -0.0092    0.3       0.49
#   -0.052     0.3       0.49
# S1_MOCAP_START_QPOS = [0.35, 1.5, 0.76]
# Default position
S1_MOCAP_START_QPOS = [0.3472, 1.314, 0.724]
# front vertical position
# S1_MOCAP_START_QPOS = [0.3472, 0.6, 0.7]
TABLE_HEIGHT = 0.5
START_ARM_POSE = []
START_ARM_POSE_SINGLE = [
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
]

XML_DIR = (
    str(pathlib.Path(__file__).parent.resolve()) + "/assets/"
)  # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = -0.8
MASTER_GRIPPER_JOINT_CLOSE = -1.65
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (
    MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (
    PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
    + MASTER_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
    + PUPPET_GRIPPER_POSITION_CLOSE
)
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
    MASTER_GRIPPER_POSITION_NORMALIZE_FN(x)
)

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (
    MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE
)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (
    PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE
)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    + MASTER_GRIPPER_JOINT_CLOSE
)
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    + PUPPET_GRIPPER_JOINT_CLOSE
)
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
    MASTER_GRIPPER_JOINT_NORMALIZE_FN(x)
)

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (
    MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE
)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (
    PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE
)

MASTER_POS2JOINT = (
    lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x)
    * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
    + MASTER_GRIPPER_JOINT_CLOSE
)
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - MASTER_GRIPPER_JOINT_CLOSE)
    / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
)
PUPPET_POS2JOINT = (
    lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x)
    * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
    + PUPPET_GRIPPER_JOINT_CLOSE
)
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - PUPPET_GRIPPER_JOINT_CLOSE)
    / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
)

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2
