source_len = 100
n_past = 100
target_len = 50
n_future = 50
n_features_h1 = (28+10+20+20)*2 
n_features_h2_B = 10*2 
n_features_h2_F=  28*2
n_features_h2_H=  (20+20)*2 
context_len = 128
n_features = (10+28+20+20)*2 
n_features_B = 10*2 
n_features_F =  28*2 
n_features_RH = 20*2
n_features_LH =  20*2 
window_size = 20


# Generic information
ANNOTATIONS_FILE = "annotations_raw.hdf5"
ANNOTATIONS_FILE_VAL="annotations_cleaned_masked.hdf5"
AVAILABLE_TASKS = ["FC1_T", "FC2_T", "FC1_A", "FC2_A", "FC1_G", "FC2_G", "FC1_L", "FC2_L"]

# Others
DISTANCES_FILENAME = "distances.csv"
GT_FILENAME = "ground_truth.json"
PRED_FILENAME = "predictions.json"

# Sessions splits
SPLIT_TRAIN = ['002003', '003005', '004096', '004115', '005013', '005134', '006007', '006153', '009106', '009167', '010011', '010034', '011034', '013041', '015126', '015133', '017027', '017149', '017150', '018020', '018025', '020025', '020090', '020149', '020150', '023102', '023191', '025044', '025157', '027076', '027113', '027118', '027154', '030078', '030079', '030082', '034035', '034121', '034133', '035040', '035114', '035166', '040090', '041083', '041084', '042084', '042092', '043079', '043143', '043152', '044156', '044157', '051076', '051100', '055125', '055128', '058059', '058110', '059134', '076122', '076143', '078156', '079123', '079142', '082174', '092096', '092107', '092108', '092168', '097098', '097142', '098126', '100116', '101116', '101139', '102173', '102176', '106108', '106148', '106173', '110136', '112113', '112132', '113132', '114166', '115175', '118125', '118154', '122144', '123188', '127128', '127129', '127184', '128129', '136175', '139140', '139189', '140170', '144169', '144171', '144176', '148151', '151164', '151165', '152153', '156169', '164165', '167168', '169171', '171172', '172185', '173176', '173179', '184185', '188189', '191192']
SPLIT_VALIDATION = ['001080', '001081', '038039', '052057', '080081', '085186', '085190', '119124', '119145', '119190', '124145', '141182', '141183', '146147', '180181', '181182', '182183', '183186']
SPLIT_TEST = ['008105', '056109', '066067', '086087', '086089', '087088', '087089', '088089', '105117', '111130', '137138']
    

# Info about the body joints
BODY_ENTITIES = ["face", "body", "left_hand", "right_hand"]

BODY_JOINTS = 10
BODY_NAMES = ["chest", "neck", "left_chest", "right_chest", 
                "left_shoulder", "right_shoulder", "left_elbow", 
                "right_elbow", "left_wrist", "right_wrist"]
LEFT_WRIST_IDX = 8
RIGHT_WRIST_IDX = 9

# Default config for evaluation
DEFAULT_CONFIG = {
    "face_percentage": 0.25,
    "face_bins": 100,
    "body_percentage": 0.5,
    "body_bins": 10,
    "hand_percentage": 0.5,
    "hand_bins": 100
}