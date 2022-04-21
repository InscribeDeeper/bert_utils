import platform

sys_info = platform.system()

if sys_info == "Darwin":
    device_type = 'cpu'
    print("My MAC system: ", sys_info)
elif sys_info == "Windows":
    print("My windows system: ", sys_info)
    device_type = 'cpu'  # due to GPU memory size
else:
    import os
    if 'COLAB_TPU_ADDR' in os.environ:
        import torch_xla.core.xla_model as xm
        device_type = xm.xla_device()
    else:
        device_type = 'cuda'

    print("My Linux system: ", sys_info)

print("using: ", device_type)
