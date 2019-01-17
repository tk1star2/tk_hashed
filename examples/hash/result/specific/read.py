import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

for filename in glob.glob("*.npy"):
    print(filename)
    arr = np.load(filename)
    print("shape is {}".format(np.shape(arr)))
    print("dimension is {}".format(np.ndim(arr)))
    #print(arr)
    #arr0 = arr[arr>0]
    if(np.ndim(arr)==1):
	"""
    	print(arr)
    	plt.plot(arr)
    	plt.show()
	"""
	continue
    elif(np.ndim(arr)==2):
	"""
    	print(arr)
    	plt.plot(arr)
    	plt.show()
	"""
	continue
    else:
	for each_in in range(arr.shape[0]):
	    for each_out in range(arr.shape[1]):
		#w, h = arr.shape[2], arr.shape[3]
		temp_arr = arr[each_in, each_out,:,:]
		max_data = np.amax(temp_arr)
		min_data = np.amin(temp_arr)

		temp_arr =((temp_arr - min_data)/(max_data - min_data))*256
		temp_arr = temp_arr.astype(np.uint8)
		#print("max_data is {} and min data is {}\n\n".format(max_data, min_data))
		#data = Image.fromarray(arr[each_in, each_out, :, :],'L')
		data = Image.fromarray(temp_arr,'L')
		data.show()
		print(temp_arr)
		break
		#plt.plot(arr[each_in, each_out, :, :])
		#plt.show()
	    break
    print('\n\n\n')
