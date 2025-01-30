#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kernel Checker
--------------
This script checks the installed kernelspecs and displays the kernel name and
display name.

Author: Azhar Muhammed
Date: July 2024
"""

# -----------------------------------------------------------------------------
# Essential Imports
# -----------------------------------------------------------------------------
import json
import os
from jupyter_client.kernelspec import KernelSpecManager

# -----------------------------------------------------------------------------
# Kernel Checking and Display
# -----------------------------------------------------------------------------
ksm = KernelSpecManager()
kernels = ksm.find_kernel_specs()

for kernel_name, kernel_path in kernels.items():
    with open(os.path.join(kernel_path, 'kernel.json')) as f:
        kernel_spec = json.load(f)
        print(f"Kernel Name: {kernel_name}, Display Name: {kernel_spec['display_name']}")
