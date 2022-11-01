#!/bin/bash

mb_pipe_path="$HOME/Mobile-Data-Processing-Pipeline"
vimrc_path="$HOME/.vimrc"

if [ "$1" == "launch_jup" ]; then
    sudo docker run -dit -p 8889:8889 -v "$mb_pipe_path":/root/Sign-Language-Recognition-Demo --name sign_recognition_demo gurudesh/copycat:copycat-cpu
    sudo docker cp "$vimrc_path" sign_recognition_demo:/root
elif [ "$1" == "run_jup" ]; then
    sudo docker exec -it sign_recognition_demo /bin/bash
elif [ "$1" == "run_command" ]; then
    if [ "$2" == "" ]; then
        echo "Must pass the command as the second argument"
        exit 1
    fi
    sudo docker exec -d sign_recognition_demo "$2"
else
    echo "Specify either launch (create the container) or run (resume the existing conainer)"
fi
