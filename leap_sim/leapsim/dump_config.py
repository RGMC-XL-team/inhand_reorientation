"""
This script dumps the config from hydra
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from leapsim.utils.reformat import omegaconf_to_dict

## PARAMETERS
# ----------------------------------------
CONFIG_SAVE_DIR = "./cfg/dict"
PRIMITIVE_NAME = "rot"      # "rot" or "flip"
CHECKPOINT_NAME = "rural-silence-171"
CONFIG_NAME = f"leap-{PRIMITIVE_NAME}-{CHECKPOINT_NAME}"
# ----------------------------------------


@hydra.main(config_path="cfg", config_name="config")
def main(config: DictConfig):
    config_dict = omegaconf_to_dict(config)
    # import pdb; pdb.set_trace()
    # with open(os.path.join(CONFIG_SAVE_DIR, f"{CONFIG_NAME}.yaml"), 'w') as f:
    #     yaml.safe_dump(config_dict, f)
    OmegaConf.save(OmegaConf.create(config_dict), os.path.join(CONFIG_SAVE_DIR, f"{CONFIG_NAME}.yaml"))


if __name__ == "__main__":
    main()
