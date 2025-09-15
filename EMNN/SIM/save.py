import os
import torch
import numpy as np
from scipy.io import savemat
from collections import OrderedDict

def get_save_dir(subfolder="models"):
    home_dir = os.path.expanduser("~")
    save_dir = os.path.join(home_dir, subfolder)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_all_models(NN_tx, SIM_tx, SIM_Rx, NN_RX, pathname_prefix="all_models"):
    save_dir = get_save_dir()
    pathname = f"EMNN/model/{pathname_prefix}.pth"
    save_path = os.path.join(save_dir, pathname)

    save_dict = {
        'NN_tx': NN_tx.state_dict(),
        'SIM_tx': SIM_tx.state_dict(),
        'SIM_Rx': [m.state_dict() for m in SIM_Rx],
        'NN_RX': [m.state_dict() for m in NN_RX]
    }
    torch.save(save_dict, save_path)
    print(f"模型已保存到: {save_path}")
    return save_path

def _remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

def load_all_models(NN_tx, SIM_tx, SIM_Rx, NN_RX, pathname_prefix="all_models", device="cpu"):
    load_dir = get_save_dir()
    pathname = f"EMNN/model/{pathname_prefix}.pth"
    load_path = os.path.join(load_dir, pathname)

    checkpoint = torch.load(load_path, map_location=device)

    NN_tx.load_state_dict(_remove_module_prefix(checkpoint['NN_tx']))
    SIM_tx.load_state_dict(_remove_module_prefix(checkpoint['SIM_tx']))
    for i, m in enumerate(SIM_Rx):
        m.load_state_dict(_remove_module_prefix(checkpoint['SIM_Rx'][i]))
    for i, m in enumerate(NN_RX):
        m.load_state_dict(_remove_module_prefix(checkpoint['NN_RX'][i]))

    print(f"模型已从 {load_path} 加载")
    return NN_tx, SIM_tx, SIM_Rx, NN_RX



def save_results_mat(args, R_error, ber, path_prefix="all_models"):
    save_dir = get_save_dir()
    pathname = f"EMNN/result/{path_prefix}.mat"
    save_path = os.path.join(save_dir, pathname)

    savemat(save_path, {
        'args': args,
        'R_error': np.array(R_error),
        'ber': ber
    })

    print(f"结果已保存到 {save_path}")