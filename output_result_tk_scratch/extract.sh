
export PYTHONPATH=/root/dataset/caffe/python:$PYTHONPATH
python output_result_tk_scratch/extract_caffe.py --model output_result_tk_scratch/lenet_train_test.prototxt --weights output_result_tk_scratch/lenet_solver_iter_6000.caffemodel --output output_result_tk_scratch/result --index 0

#export PYTHONPATH=/root/DeepCompression-caffe-after/python:$PYTHONPATH
#python extract_caffe.py --model lenet_train_test.prototxt --weights lenet_solver_iter_10000.caffemodel --output result --index 0

#export PYTHONPATH=/root/DeepCompression-caffe-after/python:$PYTHONPATH
#python examples/hash/extract_caffe.py --model examples/hash/lenet_train_test_compress_stage5.prototxt --weights examples/hash/models/lenet_finetune_stage5_iter_500.caffemodel --output examples/hash/result --index 1

#export PYTHONPATH=/root/DeepCompression-caffe-after/python:$PYTHONPATH
#python examples/hash/extract_caffe.py --model examples/hash/lenet_train_test_compress_stage10.prototxt --weights examples/hash/models/lenet_finetune_stage10_iter_1000.caffemodel --output examples/hash/result --index 2
