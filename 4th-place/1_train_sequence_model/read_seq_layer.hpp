#ifndef CAFFE_READ_SEQ_LAYER_HPP_
#define CAFFE_READ_SEQ_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/opencv.hpp>

namespace caffe {
template <typename Dtype>
class ReadSeqLayer : public Layer<Dtype> {
 public:
  explicit ReadSeqLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ReadSeq"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual
    void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

    std::string folder_;
    std::map<std::string, std::vector<std::string> > sequences_;
    std::map<std::string, std::vector<std::string> >::const_iterator pos_;
    cv::Size img_size_;
    int batch_size_;
    int sequence_size_;
    int sequence7_size_;
};

}  // namespace caffe

#endif  // CAFFE_READ_SEQ_LAYER_HPP_
