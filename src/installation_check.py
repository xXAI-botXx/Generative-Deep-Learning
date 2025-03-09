
# === lib import check ===
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, layers

import torch
from torch.nn import Module 

import mlflow

# import kaggle

# === hardware check ===
def get_hardware_info(use_in_notebook=True, install_packages=True):
    import sys
    import subprocess
    import importlib.util
    
    if install_packages:
        if importlib.util.find_spec("psutil") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
        if importlib.util.find_spec("gputil") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "gputil"], check=True)
        if importlib.util.find_spec("py-cpuinfo") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "py-cpuinfo"], check=True)

    # import needed packages
    import platform
    import psutil
    import GPUtil
    from cpuinfo import get_cpu_info

    if use_in_notebook:
        if install_packages and importlib.util.find_spec("ipython") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "ipython"], check=True)

        from IPython.display import clear_output
        clear_output()
    else:
        pass
        # os.system('cls' if os.name == 'nt' else 'clear')

    print("-"*32, "\nYour Hardware:\n")

    # General
    print("    ---> General <---")
    print("Operatingsystem:", platform.system())
    print("Version:", platform.version())
    print("Architecture:", platform.architecture())
    print("Processor:", platform.processor())

    # GPU-Information
    print("\n    ---> GPU <---")
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print("GPU Name:", gpu.name)
        print("VRAM Total:", gpu.memoryTotal, "MB")
        print("VRAM Used:", gpu.memoryUsed, "MB")
        print("Utilization:", gpu.load * 100, "%")
    try:
        import torch
        gpus = [torch.cuda.get_device_name(device_nr) for device_nr in range(torch.cuda.device_count())]
        torch_support = False
        if torch.cuda.is_available():
            torch_support = True 
            gpu_str = f"({','.join(gpus)})"
        gpu_addition = f" {gpu_str}" if torch_support else ""
        print(f"PyTorch Support: {torch_support}" + gpu_addition)
    except Exception:
        print(f"PyTorch Support: False")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        tf_support = False
        if len(gpus) > 0:
            tf_support = True 
            gpu_str = f"({','.join(gpus)})"
        gpu_addition = f" {gpu_str}" if tf_support else ""
        print(f"TensorFlow Support: {tf_support}" + gpu_addition)
    except Exception:
        print(f"TensorFlow Support: False")

    # CPU-Information
    print("\n    ---> CPU <---")
    cpu_info = get_cpu_info()
    print("CPU-Name:", cpu_info["brand_raw"])
    print("CPU Kernels:", psutil.cpu_count(logical=False))
    print("Logical CPU-Kernels:", psutil.cpu_count(logical=True))
    print("CPU-Frequence:", psutil.cpu_freq().max, "MHz")
    print("CPU-Utilization:", psutil.cpu_percent(interval=1), "%")
    

    # RAM-Information
    print("\n    ---> RAM <---")
    ram = psutil.virtual_memory()
    print("RAM Total:", ram.total // (1024**3), "GB")
    print("RAM Available:", ram.available // (1024**3), "GB")
    print("RAM-Utilization:", ram.percent, "%")

    print(f"\n{'-'*32}")



get_hardware_info(use_in_notebook=False, install_packages=False)

