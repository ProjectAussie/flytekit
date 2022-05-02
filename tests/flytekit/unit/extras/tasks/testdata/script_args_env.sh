#!/bin/bash

set -exo pipefail

echo "A: $A"
echo "B: $B"

if [ ! -z "$1" ]; then
    echo "$1"
else
    echo "Unset first positional argument"
fi

if [ ! -z "$2" ]; then
    echo "$2"
else
    echo "Unset second positional argument"
fi

SOME_VAR="This var is set"

echo "Reading SOME_VAR: ${SOME_VAR}"