#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex
pip freeze >requirements.txt
cat ./requirements.txt
