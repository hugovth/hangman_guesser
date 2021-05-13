#!/bin/bash
CMD=$1
SELF_DESTRUCT_TIME=60

set -e

# function to run passed command
function run_command
{
    sudo service filebeat start
    $CMD
}

# function to graceful destruct container
function graceful_exit
{
    echo "Container will autodestruct in $SELF_DESTRUCT_TIME seconds..."
    sleep $SELF_DESTRUCT_TIME
    echo "Shutting down container..."
}

# if we are initializing a terminal it is not worth to start filebeat
if [ "$CMD" = "bash" ]; then
    /bin/bash
else
    # Trap EXIT and ERR signals so filebeat can log final steps before container dying
    trap "graceful_exit" EXIT ERR

    # Run process
    run_command
fi
