from datasets import msraction3d, utdmhad

import config

def get_data():
    if config.DATASET == config.MSR_ACTION_3D:
        return msraction3d.get_data()
    elif config.DATASET == config.URT_MHAD:
        return utdmhad.get_data()
    else:
        raise ValueError