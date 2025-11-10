#!/bin/bash
set -e
ENV_PATH=$(poetry env info --path 2>/dev/null)

# Only add to LD_LIBRARY_PATH if it doesn't already contain the path
if [[ ! "${LD_LIBRARY_PATH}" == *"${ENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib"* ]]; then
    export LD_LIBRARY_PATH="${ENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
fi
exec "$@"