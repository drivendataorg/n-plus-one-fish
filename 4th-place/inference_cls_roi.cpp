#include <opencv2/opencv.hpp>

#include <fstream>
#include <iomanip>

#include <caffe/net.hpp>

struct Parameters
{
    Parameters()
        : modelPath(""),
          deployPath(""),
          inputFolder(""),
          shouldShow(false),
          outputFilename("")
    {}

    std::string modelPath;
    std::string deployPath;
    std::string inputFolder;
    bool shouldShow;
    std::string outputFilename;
};

struct DetectedObject
{
    float x, y, w, h;
    cv::Rect2f roi;
    float confidence;
    int classId;
    cv::Rect2f getBoundingBox() const { return roi; }
    float getConfidence() const { return confidence; }
};

struct LabeledOject
{
    enum Type {Empty = -1, Four = 0, Grey, Other, Plaice, Summer, Widow, Winter};

    LabeledOject(int id = -1, int frameId = -1, std::vector<std::string> const& tokens = std::vector<std::string>())
        : id(id),
          frameId(frameId),
          tokens(tokens)
    {
        if (tokens.size())
            length = atof(tokens[4].c_str());
        conf.resize(7, 0);
    }


    int id;
    int frameId;
    int length;
    int type;
    std::vector<float> conf;
    std::vector<std::string> tokens;
};

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

int main (int argc, char* argv[])
{
    if (argc < 7)
    {
        std::cout << "Pass parameters:" << std::endl;
        std::cout << "    -m <caffemodel file>" << std::endl;
        std::cout << "    -d <deploy file>" << std::endl;
        std::cout << "    -i <path to folder with img list>" << std::endl;
        std::cout << "    -o <output filename>" << std::endl;
        std::cout << "    -v" << std::endl;
        exit(-1);
    }
    Parameters params;
    params.modelPath = std::string(argv[2]);
    params.deployPath = std::string(argv[4]);
    params.inputFolder = std::string(argv[6]);
    if (std::string(argv[7]) == "-o")
    {
        params.outputFilename = std::string(argv[8]);
    }
    if (std::string(argv[7]) == "-v"
               || argc == 10)
    {
        params.shouldShow = true;
    }

    std::ifstream annotations(params.inputFolder);
    CV_Assert(annotations.is_open());

    std::map<std::string, std::map<int, LabeledOject> > videos;
    std::string line;
    while (std::getline(annotations, line))
    {
        std::vector<std::string> tokens = split(line, ',');
        std::string key = tokens[2];
        int frameId = atoi(tokens[1].c_str());

        int id = atof(tokens[3].c_str());
        videos[key][frameId] = LabeledOject(id, frameId, tokens);
    }

    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(0);
    google::InitGoogleLogging(argv[0]);
    caffe::Net<float> net(params.deployPath, caffe::TEST);
    net.CopyTrainedLayersFrom(params.modelPath);

    caffe::Blob<float>* dataLayer = net.input_blobs()[0];
    const int nChns = dataLayer->channels();
    const float nCols = dataLayer->width();
    const float nRows = dataLayer->height();
    const size_t batchSize = 1;
    cv::Scalar meanPixel(104, 117, 123);

    dataLayer->Reshape(batchSize, nChns, nRows, nCols);
    net.Reshape();


    caffe::Net<float> clsNet("<path_to>/deploy_roi.prototxt", caffe::TEST);
    clsNet.CopyTrainedLayersFrom("<path_to>/cls_roi.caffemodel");
    caffe::Blob<float>* clsNetDataLayer = clsNet.input_blobs()[0];
    caffe::Blob<float>* clsNetResLayer = clsNet.output_blobs()[0];

    for (auto videoIt : videos)
    {
        //std::cout << videoIt.first << std::endl;
        cv::VideoCapture cap("<path_to>/test_videos/" + videoIt.first + ".mp4");
        CV_Assert(cap.isOpened());
        int frameId = -1;
        cv::Mat img;
        while (cap.read(img))
        {
            frameId++;
            if (videoIt.second.find(frameId) == videoIt.second.end())
                continue;

            cv::Mat resizedImg;
            cv::resize(img, resizedImg, cv::Size(nCols, nRows));
            std::vector<cv::Mat> planes(3);
            cv::split(resizedImg, planes);
            for (size_t pId = 0; pId < planes.size(); pId++)
            {
                cv::Mat dst(nRows, nCols, CV_32FC1, (void*)(dataLayer->mutable_cpu_data() + dataLayer->offset(0, pId)));
                planes[pId].convertTo(dst,
                            CV_32FC1, 1, -meanPixel(pId));
                dst *= 0.017;
            }
            net.Forward();

            boost::shared_ptr<caffe::Blob<float> > detectionsLayer = net.blob_by_name("detection_out");
            const float* detectionsData = detectionsLayer->cpu_data();
            const int nDetections = detectionsLayer->height();
            cv::Mat sample;
            cv::Rect rect;
            int length = 0;
            for (int i = 0; i < std::min(1, nDetections)/*nDetections*/; i++)
            {
                const int label = detectionsData[i * 7 + 1];
                if (label == 7)
                    continue;

                rect.x = detectionsData[i * 7 + 3] * img.cols;
                rect.y = detectionsData[i * 7 + 4] * img.rows;
                rect.width = (detectionsData[i * 7 + 5] - detectionsData[i * 7 + 3]) * img.cols;
                rect.height = (detectionsData[i * 7 + 6] - detectionsData[i * 7 + 4]) * img.rows;

                if (rect.area() < 20)
                    continue;

                if (rect.height > rect.width * 1.3)
                {
                    length = rect.height;
                } else if (rect.width > rect.height * 1.3)
                {
                    length = rect.width;
                } else
                {
                    length = sqrt(rect.width * rect.width + rect.height * rect.height);
                }

                rect.x -= rect.width * 0.15;
                rect.width += rect.width * 0.3;
                rect.y -= rect.height * 0.15;
                rect.height += rect.height * 0.3;
                if (rect.width > rect.height)
                {
                    int diff = rect.width - rect.height;
                    rect.y -= diff/2;
                    rect.height += diff;
                } else if (rect.width < rect.height)
                {
                    int diff = rect.height - rect.width;
                    rect.x -= diff/2;
                    rect.width += diff;
                }

                int top = 0, bottom = 0, left = 0, right = 0;
                if (rect.x < 0)
                {
                    left -= rect.x;
                    rect.width += rect.x;
                    rect.x = 0;
                }
                if (rect.y < 0)
                {
                    top -= rect.y;
                    rect.height += rect.y;
                    rect.y = 0;
                }
                if (rect.x + rect.width > img.cols)
                {
                    int diff = rect.x + rect.width - img.cols;
                    right = diff;
                    rect.width -= diff;
                }
                if (rect.y + rect.height > img.rows)
                {
                    int diff = rect.y + rect.height - img.rows;
                    bottom = diff;
                    rect.height -= diff;
                }
                cv::copyMakeBorder(img(rect), sample, top, bottom, left, right, cv::BORDER_CONSTANT, meanPixel);
            }
            LabeledOject labeledObject = videoIt.second[frameId];
            if (length > 0)
            {
                labeledObject.length = length;
            }

            std::cout << "0," << labeledObject.frameId << "," << videoIt.first << ","
                      << labeledObject.id << "," << labeledObject.length;

            if (!sample.empty())
            {
                int detRows = clsNetDataLayer->height();
                int detCols = clsNetDataLayer->width();
                cv::Mat resizedImgDet;
                cv::resize(sample, resizedImgDet, cv::Size(detCols, detRows));
                std::vector<cv::Mat> planesDet(3);
                cv::split(resizedImgDet, planesDet);
                for (size_t pId = 0; pId < planesDet.size(); pId++)
                {
                    cv::Mat dst(detRows, detCols, CV_32FC1, (void*)(clsNetDataLayer->mutable_cpu_data() + clsNetDataLayer->offset(0, pId)));
                    planesDet[pId].convertTo(dst,
                                             CV_32FC1, 1, -meanPixel(pId)); // dont't subtract mean for resnet10, https://github.com/farmingyard/caffe-mobilenet
                    //dst *= 0.017; // uncomment for mobilenet
                }
                clsNet.Forward();

                const float* clsResData = clsNetResLayer->cpu_data();
                {
                    for (int i = 0; i < 7; i++)
                    {
                        labeledObject.conf[i] = clsResData[i];
                    }
                }
                for (int i = 0; i < 7; i++)
                {
                    std::cout << "," << labeledObject.conf[i];
                }
            } else
            {
                for (int i = 0; i < 7; i++)
                {
                    std::cout << "," << labeledObject.tokens[5 + i];
                }
            }
            std::cout << std::endl;

        }
    }


    return 0;
}
