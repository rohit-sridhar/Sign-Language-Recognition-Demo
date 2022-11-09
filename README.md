# Sign Language Recognition Demo

## Getting Started
Clone this git repository to your machine using git clone
```
git clone https://github.com/rohit-sridhar/Sign-Language-Recognition-Demo.git
```

Install [Docker](https://www.docker.com) if you haven't already and run this Docker command to download the Docker Image
```
docker pull gurudesh/copycat:popsign-cpu
```

To launch the Docker Image as a Container, first open the `run_docker.sh` file and modify the variable `mb_pipe_path`. This value should be the absolute path to the Sign Language Recognition Demo repo (this repo). This directory will be mounted within the container under `/root`.

Next, execute `run_docker.sh launch_demo`. Run `docker ps` to confirm that the container has launched with the image name `gurudesh/copycat:popsign-cpu`. You should now be able to access the container via a CLI by running `run_docker.sh run_demo`.

To stop the container, run `docker stop CONTAINER ID`, where the `CONTAINER ID` is retrieved by running `docker ps`. To remove the container from memory, run `docker rm CONTAINER ID`

## Using the Demo
Before proceeding, download the data at this Google Drive [link](https://drive.google.com/file/d/1_sImmOjPiflbV7TWDzTiHs1W1qMF3DtY), extract the videos and add them to `Sign-Language-Recognition-Demo/videos`.

Open the Docker container in your CLI (see previous section for details). Once launched, `cd` to `/root/Sign-Language-Recognition-Demo` and run the following command
```
jupyter notebook --ip 0.0.0.0 --port 8889 --allow-root --no-browser
```

The commands above assume you are running Docker from your local machine. You may need to run this on a remote server (due to limited computer resources on your local machine). To do so, follow the instructions above on the remote machine, then create an SSH tunnel from your local machine. Either use port 8889, or change the port in the command above and in the launch script, `run_docker.sh`. Here is a command to create a tunnel on from your local port 8889 to the remote machine's port 8889. This command blocks while tunneling.
```
ssh -N -L 8889:SERVER_ADDRESS:8889 USERNAME@SERVER_ADDRESS
```

You can now access the Jupyter UI by navigating to `localhost:8889` in your web browser.

## Troubleshooting
* Sometimes, you may need to relaunch the docker container, while an existing one is running. Docker won't let you launch two containers with the same name. Stop the original container with the command `docker stop CONTAINER ID` and then the command `docker rm CONTAINER ID`.

* If the `Sign-Language-Recognition-Demo` directory is empty once you launch the docker container, make sure you check the path to it in the `run_docker.sh` script. Having the incorrect path can cause issues. To fix it, remove the docker container with the empty directory and relaunch it using the script with the fixed path.

* If you run out of memory while processing Mediapipe Features, you should limit the number of threads used. This variable is set in the notebook. You may also directly download the mediapipe features [here](https://drive.google.com/file/d/1opuR5k8AwmoivuOBHePvT9_HhRYhJmT1/view?usp=sharing).
