#include <vector>

#include "caffe/layers/read_seq_layer.hpp"
#include <opencv2/opencv.hpp>

namespace caffe {

namespace
{
cv::Mat rotate(cv::Mat src, double angle)
{
    cv::Mat dst;
    cv::Point2f pt(src.cols/2., src.rows/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
    return dst;
}

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

}

template <typename Dtype>
void ReadSeqLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ReadSeqParameter param = this->layer_param_.read_seq_param();
  folder_ = param.folder();
  sequence_size_ = 11;
  sequence7_size_ = 5;

  std::ifstream file_in(std::string(param.images_file()).c_str());
  CV_Assert(file_in.is_open());
  std::ifstream labels(std::string(param.labels_file()).c_str());
  CV_Assert(labels.is_open());
  std::string line;
  std::map<std::string, bool> sequences_labels;
  while (file_in >> line)
  {
      std::vector<std::string> tokens = split(line, '_');
      std::string key = tokens[0];
      sequences_[key].push_back(line);
      if (sequences_labels.find(key) == sequences_labels.end())
      {
          sequences_labels[key] = false;
      }
      std::string label;
      labels >> label;
      sequences_labels[key] |= atoi(label.c_str());
  }
  for (std::map<std::string, bool>::const_iterator it = sequences_labels.begin();
       it != sequences_labels.end(); ++it)
  {
      sequences_[it->first].push_back(it->second ? "1" : "0");
  }
  for (std::map<string, std::vector<std::string> >::iterator it = sequences_.begin(); it != sequences_.end();)
  {
      if (it->second.size() != sequence_size_ + 1)
      {
          sequences_.erase(it++);
      } else
      {
          ++it;
      }
  }
  pos_ = sequences_.begin();
  img_size_ = cv::Size(400, 400);
  batch_size_ =  atoi(param.batch_size().c_str());
}

template <typename Dtype>
void ReadSeqLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::vector<int> shape(4);
  shape[0] = batch_size_ * sequence7_size_;
  shape[1] = 3;
  shape[2] = img_size_.height;
  shape[3] = img_size_.width;
  top[0]->Reshape(shape);

  shape.resize(2);
  shape[0] = batch_size_;
  shape[1] = sequence7_size_;
  top[1]->Reshape(shape);
  top[2]->Reshape(shape);
}

template <typename Dtype>
void ReadSeqLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  cv::Scalar meanPixel(104, 117, 123);

  for (int bId = 0; bId < batch_size_; bId++, pos_++)
  {
      if (pos_ == sequences_.end())
      {
          pos_ = sequences_.begin();
      }

      std::vector<std::string> img_names = pos_->second;
      int label = img_names[sequence_size_] == "1" ? 1 : 0;

      int shift = (sequence_size_ - sequence7_size_) / 2;
      bool shouldRotate = rand() % 2;
      for (int i = 0; i < sequence7_size_; i++)
      {
          cv::Mat img = cv::imread(folder_ + "/" + img_names[i + shift], CV_LOAD_IMAGE_COLOR);
          CV_Assert(!img.empty());
          cv::resize(img, img, img_size_);
          if (shouldRotate)
          {
              img = rotate(img, 90);
          }

          std::vector<cv::Mat> planes(3);
          cv::split(img, planes);
          for (size_t pId = 0; pId < planes.size(); pId++)
          {
              planes[pId].convertTo(
                          cv::Mat(img_size_.height, img_size_.width, CV_32FC1,
                                  (void*)(top[0]->mutable_cpu_data() + top[0]->offset(bId * sequence7_size_ + i, pId))),
                      CV_32FC1, 1, -meanPixel(pId));
          }
          *(top[1]->mutable_cpu_data() + bId * sequence7_size_ + i) = (i == 0) ? 0 : 1;
          *(top[2]->mutable_cpu_data() + bId * sequence7_size_ + i) = label;
      }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ReadSeqLayer);
#endif

INSTANTIATE_CLASS(ReadSeqLayer);
REGISTER_LAYER_CLASS(ReadSeq);

}  // namespace caffe
