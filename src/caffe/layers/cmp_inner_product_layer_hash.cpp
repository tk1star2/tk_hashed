#include <vector>
#include <iostream>
using namespace std ;

#include "caffe/filler.hpp"
#include "caffe/layers/cmp_inner_product_layer_hash.hpp"
//caffe_cpu_gemm() function
#include "caffe/util/math_functions.hpp"
#include "caffe/kmeans.hpp"
#include "caffe/xxhash.hpp"

//#define my_caffe_hash
#define my_caffe_DC
//#define my_caffe_inner
namespace caffe {

template <typename Dtype>
void CmpInnerProductHashLayer<Dtype>::ComputeBlobMask()
{
  printf("\n\n\nERROR!!!!!! this must not be called!!!!\n\n\n");
  int count = this->blobs()[0]->count();

  //calculate min max value of weight
  const Dtype* weight = this->blobs_[0]->cpu_data();
  int *mask_data = this->masks_.mutable_cpu_data();
  vector<Dtype> sort_weight(count);
  for (int i = 0; i < count; ++i)
  {
     sort_weight[i] = fabs(weight[i]);
  }

  sort(sort_weight.begin(), sort_weight.end());
  
  float ratio = this->sparse_ratio_;
  int index = int(count*ratio);
  Dtype thr ;
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  float rat = 0;
  if(index >0) {
    thr = sort_weight[index - 1];
    LOG(INFO) << "THR: "<<thr<<endl ;

    for (int i = 0; i < count; ++i)
    {
      mask_data[i] =  ((weight[i] > thr || weight[i] < -thr) ? 1 : 0) ;
      muweight[i] *= mask_data[i];
      rat += (1-mask_data[i]) ;
    }
  } else {
      for (int i = 0; i < count; ++i)
      {    
         mask_data[i]  = (weight[i]== 0 ? 0 : 1);
         rat += (1-mask_data[i]) ;
      } 
  }
  LOG(INFO) << "sparsity: "<< rat/count <<endl;

  if(this->quantize_term_)
  {
    int nCentroid = this->class_num_;
    kmeans_cluster(this->indices_.mutable_cpu_data(), this->centroids_.mutable_cpu_data(), muweight, count, mask_data/*this->masks_*/, /*max_weight, min_weight,*/ nCentroid, 1000);
  }
}



template <typename Dtype>
void CmpInnerProductHashLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights with given filler
    printf("\n\n\n**********************here*********************** \n\n\n");
    printf("filler OK\n");
    printf("\n\n\n**********************here*********************** \n\n\n");
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));

      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);


#if defined(my_caffe_DC) || defined(my_caffe_hash)
  // this is why we can call immediately
  this->sparse_ratio_ = this->layer_param_.inner_product_param().sparse_ratio();
  this->class_num_ = this->layer_param_.inner_product_param().class_num();
  this->quantize_term_ = this->layer_param_.inner_product_param().quantize_term();
  int count = this->blobs_[0]->count() ; 

  // initialize mask matrix to 1
  vector<int> mask_shape(1,count);
  this->masks_.Reshape(mask_shape);
  int *mask_data = this->masks_.mutable_cpu_data();
  caffe_set(count, 1, this->masks_.mutable_cpu_data());

  if(quantize_term_)
  {   
    this->indices_.Reshape(mask_shape);
    vector<int> cen_shape(1,class_num_);
    this->centroids_.Reshape(cen_shape);
    this->tmpDiff_.Reshape(cen_shape);
    this->freq_.Reshape(cen_shape);
   
  //TK
  //std::cout << "weight filler  is "<< this->layer_param_.inner_product_param().weight_filler() << std::endl;
  //printf("weight filler  is %s",this->layer_param_.inner_product_param().weight_filler());


/*
  int nCluster = this->class_num_;
  Dtype *cCentro = this->centroids_.mutable_cpu_data();
  for(int k = 0; k < nCluster; ++k) {
  	cCentro[k] = k/nCluster;
  }
*/
  } 
#endif
}

template <typename Dtype>
void CmpInnerProductHashLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void CmpInnerProductHashLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
#if defined(my_caffe_DC) || defined(my_caffe_hash)
				//CAN CHANGE WEIGHT : MUWEIGHT
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  const int *mask_data = this->masks_.cpu_data();
  int count = this->blobs_[0]->count();

  //PRUNING
  for (int i = 0; i < count; ++i){
    muweight[i] *= mask_data[i];
    /*
    std::array<int, 4> input {i, i+1, i+2, i+3};
    xxh::hash_t<32> hash = xxh::xxhash<32>(input);
    printf("\n\n\ncount is %d\n", count);
    printf("i is %d\n", i);
    int index_a = bottom[0]->count(1);
    printf("size input index is %d\n", index_a);
    printf("index1 is %d\n", i%index_a);
    printf("index2 is %d\n", (int)(i/index_a));
    printf("XXH result is %lu\n", hash);
    printf("XXH modified is %lu\n", (hash%(count/80)));
    */
    
  }
  //QUANTIZATION : HERE!!
  if(this->quantize_term_)
  {
    const Dtype *cent_data = this->centroids_.cpu_data();
    // take indice matrix! this is Deep compression!
    const int *indice_data = this->indices_.cpu_data();

    for (int i = 0; i < count; ++i)
    {
       if (mask_data[i]){
	 //HERE!!
#ifdef my_caffe_hash
         int index_a = bottom[0]->count(1);
         int index1 = i%index_a;
         int index2 = (int)(i/index_a);
    	 std::array<int, 6> input {index1, index1+1, index1+2, index2, index2+1, index2+2};
    	 xxh::hash_t<32> hash = xxh::xxhash<32>(input);
	 muweight[i] = cent_data[hash % (count/80)];
#else
         muweight[i] = cent_data[indice_data[i]];
#endif
       }
    }
  }
#endif
  // LET'S COMPLETE INNER PRODUCT
  const Dtype* bottom_data = bottom[0]->cpu_data();
			  //CAN CHANGE DATA : TOP_DATA
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);

  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void CmpInnerProductHashLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    //printf("\n\n\n**********************here*********************** \n\n\n");
    //printf("what is param_propagate_down_[0] : %d\n", this->param_propagate_down_[0] );
    //std::cout << "param is "<< this->param_propagate_down_[0] << std::endl;
    //printf("\n\n\n**********************here*********************** \n\n\n");
  
    //printf("\n\n\n-------------------------B--------------------------------- \n\n\n");
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  
#if defined(my_caffe_DC) || defined(my_caffe_hash)
  int count = this->blobs_[0]->count();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  const int *mask_data = this->masks_.cpu_data();

  //START!!
  //PRUINING
  for (int j = 0; j < count; ++j)
    weight_diff[j] *= mask_data[j];
  
  //QUANTIZATION:HERE!!!*******************************
  if(this->quantize_term_)
  {
    //centroid vector!!
    const int *indice_data = this->indices_.cpu_data();
    vector<Dtype> tmpDiff(this->class_num_);
    vector<int> freq(this->class_num_);

    //TK start here
    //accumulate gradients in temp_centroid
    for (int j = 0; j < count; ++j)
    {
       if (mask_data[j])
       {
#ifdef my_caffe_hash
          int index_a = bottom[0]->count(1);
          int index1 = j%index_a;
          int index2 = (int)(j/index_a);
    	  std::array<int, 6> input {index1, index1+1, index1+2, index2, index2+1, index2+2};
    	  xxh::hash_t<32> hash = xxh::xxhash<32>(input);
	  tmpDiff[hash%(count/80)] += weight_diff[j];
          freq[hash%(count/80)]++;
#else
          tmpDiff[indice_data[j]] += weight_diff[j];
          freq[indice_data[j]]++;
#endif
       }
    }

    for (int j = 0; j < count; ++j)
    {
       if (mask_data[j]){
#ifdef my_caffe_hash
          int index_a = bottom[0]->count(1);
          int index1 = j%index_a;
          int index2 = (int)(j/index_a);
    	  std::array<int, 6> input {index1, index1+1, index1+2, index2, index2+1, index2+2};
    	  xxh::hash_t<32> hash = xxh::xxhash<32>(input);
          weight_diff[j] = tmpDiff[hash%(count/80)] / freq[hash%(count/80)];
#else
          weight_diff[j] = tmpDiff[indice_data[j]] / freq[indice_data[j]];
#endif
       }
    }
  }
  // END***********************************************
#endif
  }


  // weight_diff[] will be applied to weight!!!!!!!!!!!!1
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CmpInnerProductHashLayer);
#endif

INSTANTIATE_CLASS(CmpInnerProductHashLayer);
REGISTER_LAYER_CLASS(CmpInnerProductHash);

}  // namespace caffe
