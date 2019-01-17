#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=models/lenet_tk/lenet_solver.prototxt $@
