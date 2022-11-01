# Sign Language Recognition Demo

## Getting Started
Clone this git repository to your machine using git clone
```
git clone https://github.com/rohit-sridhar/Sign-Language-Recognition-Demo.git
```

Install [Docker](https://www.docker.com) if you haven't already and run this Docker command to download the Docker Image
```
docker pull gurudesh/copycat:copycat-cpu
```

To launch the Docker Image as a Container, first open the `run_docker.sh` file and modify the variable `mb_pipe_path`. This value should be the absolute path to the Sign Language Recognition Demo repo (this repo). This directory will be mounted within the container under `/root`.

Next, execute `run_docker.sh launch_demo`. Run `docker ps` to confirm that the container has launched with the image name `gurudesh/copycat:copycat-cpu`. You should now be able to access the container via a CLI by running `run_docker.sh run_demo`.

To stop the container, run `docker stop CONTAINER ID`, where the `CONTAINER ID` is retrieved by running `docker ps`. To remove the container from memory, run `docker rm CONTAINER ID`

## Using the Demo
Before proceeding, download the data at this Google Drive [link](https://drive.google.com/file/d/1_sImmOjPiflbV7TWDzTiHs1W1qMF3DtY), extract the videos and add them to `Sign-Language-Recognition-Demo/videos`.

Next, install jupyterlab and ipywidgets (this won't be necessary after we get a new Docker Image). Run the following commands:
```
pip install jupyterlab
pip install ipywidgets
```

Next, open the Docker container in your CLI (see previous section for details). Once launched, `cd` to `/root/Sign-Language-Recognition-Demo` and run the following command
```
jupyter notebook --ip 0.0.0.0 --port 8889 --allow-root --no-browser
```

You may need to install jupyter, depending on the Docker container you are using. Run `pip install jupyterlab` to install it. Once launched, open `sign_recognition_demo.ipynb` to run the demo.

Note that if you're running this on a remote server, you will first have to set up a tunnel to that server which tunnels to the port that jupyter maps the docker ports to.
