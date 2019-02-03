
#./build/tools/caffe test --model models/lenet/lenet_train_test.prototxt --weights output_result/lenet_solver_iter_10000.caffemodel
#./build/tools/caffe test --model models/lenet_tk1/lenet_train_test.prototxt --weights output_result_tk/lenet_solver_iter_10000.caffemodel
./build/tools/caffe test --model models/lenet_tk/lenet_train_test.prototxt --weights output_result_tk_scratch/lenet_solver_iter_1000.caffemodel
