import json
import os

import numpy as np


def load_species_config(n, config_path=None):
    """
    读取物种参数与竞争系数矩阵，并按 n 截取。
    n 范围：2-10
    """
    if not (2 <= n <= 10):
        raise ValueError("n 必须在 2-10 之间")

    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "species_params.json")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    species_all = data["species"]
    alpha_all = np.array(data["alpha"], dtype=float)

    if n > len(species_all):
        raise ValueError("n 超过配置文件中的物种数量")

    species_params = species_all[:n]
    alpha = alpha_all[:n, :n]
    N0 = np.array([sp["N0"] for sp in species_params], dtype=float)

    return species_params, alpha, N0
