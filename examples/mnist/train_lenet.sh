#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=models/lenet/lenet_solver.prototxt $@
