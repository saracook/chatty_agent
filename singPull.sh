#!/bin/bash

# Variables
IMAGE_NAME="bcwrit/chat:latest"
SIF_NAME="chat.sif"
MODEL_PATH="/oak/stanford/groups/ruthm/bcritt/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"
DATABASE_FILE=".langchain.db"
SCRIPT_PATH="chattyCaching.py"  # Proper inline scripting paths

# Clean any existing Singularity images
#rm -f $SIF_NAME

# Verify database file exists ensuring correct permissions
echo "Ensuring database file exists with right permissions..."
if [ ! -f $DATABASE_FILE ]; then
    touch $DATABASE_FILE
fi
chmod 666 $DATABASE_FILE

# Verify model path exists ensuring correct permissions semantics
echo "Ensuring model path exists with right permissions..."
if [ ! -d $MODEL_PATH ]; then
    echo "Model path does not exist: $MODEL_PATH"
    exit 1
fi
chmod -R 755 $MODEL_PATH

# Clean Singularity cache ensuring fresh enforced pull
#echo "Cleaning Singularity cache for enforcing fresh pull..."
#singularity cache clean --force

# Pull Docker image convert runtime singularity enforce pull
#echo "Pull Docker image, convert Singularity ensuring enforcement..."
#singularity pull --name $SIF_NAME docker://$IMAGE_NAME

# Create runtime directory ensuring firewall checkpoints verifying container state paths aligned
#echo "Creating necessary directories if aligned container paths checkpoints..."
#singularity exec $SIF_NAME mkdir -p /app/model  # Model dynamically bound

# Debugging auxiliary confirming checks ensuring directory paths
singularity exec $SIF_NAME ls -lha /app
singularity exec $SIF_NAME ls -lha /app/Documents

# Running Singularity ensuring correct directory binding noting verifying aligned paths correctly
echo "Running singularity affirming ensuring integrity centered bindings..."
singularity exec --nv \
    --bind $MODEL_PATH:/app/model \
    --bind $DATABASE_FILE:/app/.langchain.db \
    --bind $SCRIPT_PATH:/app/chattyCaching.py \
    $SIF_NAME /bin/bash -c "source /opt/conda/bin/activate chatbot && python3 /app/chattyCaching.py"
