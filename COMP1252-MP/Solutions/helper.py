#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper Functions
----------------
This script contains helper functions for the Earth Data API. It includes
functions for setting up logging, checking environment variables, and
formatting search results.

Author: Azhar Muhammed
Date: July 2024
"""

# -----------------------------------------------------------------------------
# Essential Imports
# -----------------------------------------------------------------------------
import os
import sys
import torch
import logging
from dotenv import load_dotenv
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from colorlog import ColoredFormatter

# -----------------------------------------------------------------------------
# Logging Setup Function
# -----------------------------------------------------------------------------
def setup_logging(log_level=logging.INFO, file=('default', 'None')):
    # Create logs directory if it doesn't exist
    log_dir = '../Docs/Logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure rich console with custom theme
    custom_theme = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "critical": "red reverse",
        "debug": "green",
    })
    console = Console(theme=custom_theme)

    # File formatter (without colors)
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Clear any existing handlers from the root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    # Configure rich handler for enhanced console output
    rich_handler = RichHandler(
        console=console,
        enable_link_path=True,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True
    )
    
    # File handler for root logger
    file_handler = logging.FileHandler(os.path.join(log_dir, 'processing.log'))
    file_handler.setFormatter(file_formatter)

    # Add handlers to root logger
    root_logger.addHandler(rich_handler)
    root_logger.addHandler(file_handler)

    # Module-specific logger
    logger = logging.getLogger(file)
    logger.setLevel(log_level)
    # Don't add handlers to module logger - it will use root logger's handlers

    # File-only logger (if you need a separate logger that only writes to file)
    file_logger = logging.getLogger('file_only')
    file_logger.setLevel(log_level)
    file_logger.propagate = False  # Don't propagate to root logger
    file_only_handler = logging.FileHandler(os.path.join(log_dir, 'processing.log'))
    file_only_handler.setFormatter(file_formatter)
    file_logger.addHandler(file_only_handler)

    if file not in [None, 'None']:
        logger.info(f"Running {file} script...")

    return logger, file_logger

logger, file_logger = setup_logging(file=None)

# -----------------------------------------------------------------------------
# Environment Variable Check Function
# -----------------------------------------------------------------------------
def env_check(var_name, placeholder):
    """
    Check if the environment variable is correctly set.

    Args:
        var_name (str): Name of the environment variable.
        placeholder (str): Default placeholder value to check against.

    Returns:
        str: Value of the environment variable.

    Raises:
        SystemExit: If the variable is not set or set to the default placeholder.
    """
    value = os.getenv(var_name)
    if not value or value == placeholder:
        logging.error(f"Error: {var_name} is not set or is set to the default placeholder. Please update your .env file.")
        sys.exit(1)
    return value

# -----------------------------------------------------------------------------
# Device Information Function
# -----------------------------------------------------------------------------
def get_device(pretty='silent'):
    """
    Get PyTorch device information with optional output control.
    Args:
        pretty (str): Output mode - 'print', 'log', or 'silent' (default)
    Returns:
        str: Device type ('mps', 'cuda', or 'cpu')
    """
    device_info = []
    device_info.append(f"PyTorch version: {torch.__version__}")

    # Running on a local machine
    if torch.backends.mps.is_available():
        device = 'mps'
        message = "Apple Silicon Metal Performance Shader (MPS) Support"
        device_info.extend([
            f"\n{message}",
            f"{'-' * len(message)}",
            f"Apple MPS built status : {torch.backends.mps.is_built()}",
            f"Apple MPS availability : {torch.backends.mps.is_available()}",
            f"{'-' * len(message)}"
        ])
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    device_info.append(f"Using device: {device.upper()}\n")

    # Handle output based on pretty parameter
    if pretty == 'print':
        for line in device_info:
            print(line)
    elif pretty == 'log':
        for line in device_info:
            logger.info(line)

    return device

# -----------------------------------------------------------------------------
# Results Formatting Function
# -----------------------------------------------------------------------------
def format(results):
    formatted_results = []

    for idx, result in enumerate(results, start=1):
        # Use .get() to safely access dictionary keys
        collection = result.get('Collection', {})
        collection_title = collection.get('EntryTitle', 'No Title')

        # Safely handle spatial coverage
        spatial_coverage = result.get('Spatial coverage', {}).get('HorizontalSpatialDomain', {}).get('Geometry', {}).get('GPolygons', [])
        if spatial_coverage:
            boundary_points = spatial_coverage[0].get('Boundary', {}).get('Points', [])
        else:
            boundary_points = 'No spatial coverage available'

        # Safely handle temporal coverage
        temporal_coverage = result.get('Temporal coverage', {}).get('SingleDateTime', 'No Temporal Data')

        # Handle size
        size = result.get('Size(MB)', 'Unknown Size')

        # Handle data URLs
        data_urls = result.get('Data', [])

        # Formatting each search result
        formatted_results.append(f"Result {idx}:\n"
                                 f"Title: {collection_title}\n"
                                 f"Spatial Coverage (Boundary Points): {boundary_points}\n"
                                 f"Temporal Coverage: {temporal_coverage}\n"
                                 f"Size: {size} MB\n"
                                 f"Data URLs:\n" + "\n".join(f" - {url}" for url in data_urls) + "\n")

    return "\n".join(formatted_results)

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
load_dotenv("../Keys/.env")
