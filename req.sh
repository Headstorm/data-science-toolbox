#!/bin/bash

for d in */ ; do
    echo "$d"
    cd $d
    pipreqs .
    cd -
done
