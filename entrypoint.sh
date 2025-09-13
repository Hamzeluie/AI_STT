#!/bin/bash
ENV_PATH=$(pwd)

# Only add to LD_LIBRARY_PATH if it doesn't already contain the path
if [[ ! "${LD_LIBRARY_PATH}" == *"${ENV_PATH}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"* ]]; then
    export LD_LIBRARY_PATH="${ENV_PATH}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
fi
exec "$@"