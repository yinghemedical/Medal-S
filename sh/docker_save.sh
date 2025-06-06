#!/bin/bash
TEAM_NAME="teamx"

# Save the Docker image to a compressed file
echo "Saving Docker image as ${TEAM_NAME}_alldata.tar.gz..."
docker save ${TEAM_NAME}:latest | gzip > ${TEAM_NAME}_alldata.tar.gz

# Check if save succeeded
if [ $? -ne 0 ]; then
    echo "Error: Docker save failed"
    exit 1
fi

echo "Docker image saved successfully as ${TEAM_NAME}_alldata.tar.gz"