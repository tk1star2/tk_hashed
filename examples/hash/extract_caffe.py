import caffe
import numpy as np
import argparse
import os

#export PYTHONPATH=/root/dataset/caffe/python:$PYTHONPATH
#python extract_caffe.py 
#--model lenet_train_test.prototxt 
#--weights lenet_solver_iter_10000.caffemodel 
#--output result 
#--index 1

def extract_caffe_model(model, weights, output_path, index):
  """extract caffe model's parameters to numpy array, and write them to files
  Args:
    model: path of '.prototxt'
    weights: path of '.caffemodel'
    output_path: output path of numpy params 
  Returns:
    None
  """
  net = caffe.Net(model, caffe.TEST)
  net.copy_from(weights)

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  with open(output_path+"/result"+index+".txt", 'w') as f:
  	for item in net.params.items():
    		name, layer = item
    		print('convert layer: ' + name)
		#f.write("\n\n***********name: {} , layer: {}, type: {}\n\n\n".format(name, layer, net.layer[name].type));
		f.write("\n\n***********name: {} \n\n\n".format(name));
    		num = 0
    		for p in net.params[name]:
			f.write("\n\n***********size: {} \n\n\n".format(p.data.shape));
      			np.save(output_path + '/specific/' + str(name) + '_' + str(num), p.data)
			f.write("{}.variable is \n{}\n".format(num, p.data));		
      			num += 1
			
			#test
  			#with open(output_path+"/test"+index+".txt", 'w') as f2:
			#	for blob in range(0, len(p.blobs)):
			#		f.write("{}.variable is \n{}\n".format(num+1, p.blobs[blob].data));	
			#f2.close();
  f.close();

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", help="model prototxt path .prototxt")
  parser.add_argument("--weights", help="caffe model weights path .caffemodel")
  parser.add_argument("--output", help="output path")
  parser.add_argument("--index", help="whatever want")
  args = parser.parse_args()
  extract_caffe_model(args.model, args.weights, args.output,args.index)
