#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=models/lenet_tk1/lenet_solver.prototxt $@
