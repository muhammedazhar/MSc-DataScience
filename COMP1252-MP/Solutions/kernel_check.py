import json
import os
from jupyter_client.kernelspec import KernelSpecManager

ksm = KernelSpecManager()
kernels = ksm.find_kernel_specs()

for kernel_name, kernel_path in kernels.items():
    with open(os.path.join(kernel_path, 'kernel.json')) as f:
        kernel_spec = json.load(f)
        print(f"Kernel Name: {kernel_name}, Display Name: {kernel_spec['display_name']}")