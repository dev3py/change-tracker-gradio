#!/bin/bash

# Global Variables
CONTAINER_NAME="ctgr-container"
IMAGE_NAME="ctgr"
GIT_REPO_PATH="/root/change-tracker-gradio"  # Change this to the path of your repository
HOST_PORT=7866
CONTAINER_PORT=7866

# Function to stop and remove Docker container
stop_and_remove_container() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        printf "Stopping container: %s\n" "$CONTAINER_NAME"
        if ! docker stop "$CONTAINER_NAME"; then
            printf "Error: Failed to stop container %s\n" "$CONTAINER_NAME" >&2
            return 1
        fi

        printf "Removing container: %s\n" "$CONTAINER_NAME"
        if ! docker rm "$CONTAINER_NAME"; then
            printf "Error: Failed to remove container %s\n" "$CONTAINER_NAME" >&2
            return 1
        fi
    else
        printf "Container %s does not exist, skipping stop and remove.\n" "$CONTAINER_NAME"
    fi
}

# Function to pull the latest code from Git repository
pull_latest_code() {
    printf "Changing to Git repository path: %s\n" "$GIT_REPO_PATH"
    if ! cd "$GIT_REPO_PATH"; then
        printf "Error: Failed to change directory to %s\n" "$GIT_REPO_PATH" >&2
        return 1
    fi

    printf "Pulling latest code from Git...\n"
    if ! git pull; then
        printf "Error: Git pull failed\n" >&2
        return 1
    fi
}

# Function to build Docker image
build_docker_image() {
    printf "Building Docker image: %s\n" "$IMAGE_NAME"
    if ! docker build -t "$IMAGE_NAME" .; then
        printf "Error: Failed to build Docker image %s\n" "$IMAGE_NAME" >&2
        return 1
    fi
}

# Function to run Docker container
run_docker_container() {
    printf "Running Docker container: %s\n" "$CONTAINER_NAME"
    if ! docker run -p "$HOST_PORT":"$CONTAINER_PORT" --name "$CONTAINER_NAME" -d "$IMAGE_NAME"; then
        printf "Error: Failed to run Docker container %s\n" "$CONTAINER_NAME" >&2
        return 1
    fi
}

# Main Function
main() {
    stop_and_remove_container || return 1
    pull_latest_code || return 1
    build_docker_image || return 1
    run_docker_container || return 1

    printf "Deployment completed successfully.\n"
}

# Execute main function
main
