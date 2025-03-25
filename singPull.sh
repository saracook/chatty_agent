#!/bin/bash
rm chatenv.sif 
singularity pull chatenv.sif docker://bcwrit/chatenv:latest
singularity exec --nv --bind /oak/stanford/groups/ruthm/bcritt/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db:/app/model ./chatenv.sif conda run -n docker_chat python3 /app/chattyAgent.py
