#!/bin/bash

set -exo pipefail

tmp_dir=${WORK_DIR:-"/tmp"}
mkdir -p $tmp_dir
cd $tmp_dir

echo "A: $A" >> test_output.txt
echo "B: $B" >> test_output.txt

if [ ! -z $1 ]; then
    echo $1 >> test_output.txt
else
    echo "Unset first positional argument"
fi

if [ ! -z $2 ]; then
    echo $2 >> test_output.txt
else
    echo "Unset second positional argument"
fi

SOME_VAR="This var is set"

echo "Reading SOME_VAR: ${SOME_VAR}" >> test_output.txt
