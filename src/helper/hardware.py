"""
Hardware helper file.

Use me:

>>> import sys
>>> sys.path += ["./src/helper"]
>>> from hardware import get_hardware_info
>>> get_hardware_info(use_in_notebook=True, install_packages=True)

Or copy my code to colab 🥳🚀
"""



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


# testing
if __name__ == '__main__':
    get_hardware_info(use_in_notebook=True, install_packages=True)



