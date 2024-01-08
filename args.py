'''
ARGUMENTS FORMAT:

1 - DATASET FROM SINGLE MODALITY
{
    "name": "dataset_name",
    "input_path": "dataset_path",
    "output_path": "output_path",
    "mode": "train",
    "shape": [256, 256],
    "mods": ["blur", "noise", "rotate", "flip", "crop", null],
}

2 - DATASET FROM FUSED MODALITIES
{
    "name": "dataset_name",
    "input_path": ["dataset_path_1", "dataset_path_2"],
    "output_path": "output_path",
    "mode": "train",
    "shape": [256, 256,3],
    "mods": [
                ["blur", "noise", "rotate", "flip", "crop"] or None,
                ["blur", "noise", "rotate", "flip", "crop"] or None
            ],
}
'''

args = {
    "input_path": "./data/train/l_pty/",
    "output_path": "./data/train/l_pty_deneme/",
    "shape": [256, 256,3],
    "mods": ['blur', 'crop', 'flip', 'rotate', 'noise'],
}