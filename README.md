# Sign Language Recognition Demo

## Getting Started
Pull this git repository using git clone:
`git clone https://github.com/rohit-sridhar/Sign-Language-Recognition-Demo.git`

Install Docker if you haven't already and run this Docker command to download the Docker Image
`docker pull gurudesh/copycat:copycat-cpu`

To launch the Docker Image as a Container, first open the `run_docker.sh` file and modify the variable `mb_pipe_path`. This value should be the absolute path to the Sign Language Recognition Demo repo (this repo). This directory will be mounted within the container under `/root`. Next, execute the command `run_docker.sh launch_jup`. Run `docker ps` to confirm that the container is running with the image name `gurudesh/copycat:copycat-cpu`. You can now access the container via a CLI by running `run_docker.sh run_jup`.

## Launching the Jupyter Notebook
