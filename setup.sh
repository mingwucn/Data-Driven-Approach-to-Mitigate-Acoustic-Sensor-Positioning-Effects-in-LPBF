#!/bin/bash

ENV_NAME="ai"
TARGET_VER="3.9"
RECREATE=0

echo "--- Checking Conda Environment: $ENV_NAME ---"

# 1. Check if environment exists
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "✔ Environment '$ENV_NAME' found."
    
    # 2. Check Python version inside the environment
    # We use 'conda run' to execute python --version inside the env without activating it
    CURRENT_VER=$(conda run -n $ENV_NAME python --version 2>&1 | awk '{print $2}')
    
    if [[ $CURRENT_VER == $TARGET_VER* ]]; then
        echo "✔ Python version is $CURRENT_VER (Matches $TARGET_VER)."
    else
        echo "✘ Python version mismatch! Found $CURRENT_VER, but need $TARGET_VER."
        RECREATE=1
    fi
else
    echo "✘ Environment '$ENV_NAME' does not exist."
    RECREATE=1
fi

# 3. Recreate Logic
if [ $RECREATE -eq 1 ]; then
    echo "--- Starting Recreation Process ---"
    
    # Remove if it exists (to clear out bad versions or broken files)
    if conda info --envs | grep -q "^$ENV_NAME "; then
        echo "Removing existing '$ENV_NAME' environment..."
        conda remove -n $ENV_NAME --all -y
    fi
    
    # Create new
    echo "Creating new environment '$ENV_NAME' with Python $TARGET_VER..."
    conda create -n $ENV_NAME python=$TARGET_VER ipykernel -y
    
    # Install dependencies
    # We use 'conda run' to pip install so we don't have to struggle with shell activation in script
    if [ -f "requirements.txt" ]; then
        echo "Installing requirements from requirements.txt..."
        conda run -n $ENV_NAME pip install -r requirements.txt
    else
        echo "Warning: requirements.txt not found."
    fi

    # Register Kernel
    echo "Registering IPyKernel..."
    conda run -n $ENV_NAME python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"
    
    echo "--- Recreation Complete! ---"
else
    echo "--- Environment is healthy. No changes made. ---"
fi