#!/bin/bash

# Copy the following content to the system '.zshrc' file and please remove the echo command.
#-------------------------
# Get the path to the Python interpreter of the desired conda environment
ENV_PYTHON_PATH=$(which python)
# echo $ENV_PYTHON_PATH

# Extract the environment name and Python version from the path
ENV_NAME=$(dirname "$(dirname "$ENV_PYTHON_PATH")")
PYTHON_VERSION=$(python --version | awk '{split($2, a, "."); print "python" a[1]"."a[2]}')
# echo $ENV_NAME

# Construct the path to the site-packages directory
SITE_PACKAGES_PATH="$ENV_NAME/lib/$PYTHON_VERSION/site-packages"

# Export the path to the site-packages directory
export PYTHON_SITE_PACKAGES=$SITE_PACKAGES_PATH

# Print the current Python Virtual Environment path
# echo $PYTHON_SITE_PACKAGES
#-------------------------