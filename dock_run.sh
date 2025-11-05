docker run -it --gpus all \
   --shm-size 100g \
   --net=host \
   --name rlinf \
   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
   rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 /bin/bash