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

void clipRect(cv::Mat const& img, cv::Rect& roi)
{
    if (roi.x < 0)
    {
        roi.width += roi.x;
        roi.x = 0;
    }
    if (roi.y < 0)
    {
        roi.height += roi.y;
        roi.y = 0;
    }
    if (roi.x + roi.width > img.cols)
    {
        int diff = roi.x + roi.width - img.cols + 1;
        roi.width -= diff;
    }
    if (roi.y + roi.height > img.rows)
    {
        int diff = roi.y + roi.height - img.rows + 1;
        roi.height -= diff;
    }
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

    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(0);
    google::InitGoogleLogging(argv[0]);

    caffe::Net<float> det_net(params.deployPath, caffe::TEST);
    det_net.CopyTrainedLayersFrom(params.modelPath);
    caffe::Blob<float>* detNetDataLayer = det_net.input_blobs()[0];
    caffe::Blob<float>* detNetResLayer = det_net.output_blobs()[0];

    cv::Scalar meanPixel(104, 117, 123);
    int delay = 0;
    std::ifstream imgList(params.inputFolder + "/imgs.txt");
    CV_Assert(imgList.is_open());
    size_t rowCounter = 0;
    std::string filename;
    while (imgList >> filename)
    {
        cv::Mat img = cv::imread(params.inputFolder + "/" + filename + ".jpg");
        CV_Assert(!img.empty());
        std::vector<std::vector<DetectedObject> > detectedObjects;

        int detRows = detNetDataLayer->height();
        int detCols = detNetDataLayer->width();
        cv::Mat resizedImgDet;
        cv::resize(img, resizedImgDet, cv::Size(detCols, detRows));
        std::vector<cv::Mat> planesDet(3);
        cv::split(resizedImgDet, planesDet);
        for (size_t pId = 0; pId < planesDet.size(); pId++)
        {
            cv::Mat dst(detRows, detCols, CV_32FC1, (void*)(detNetDataLayer->mutable_cpu_data() + detNetDataLayer->offset(0, pId)));
            planesDet[pId].convertTo(dst,
                                     CV_32FC1, 1, -meanPixel(pId));
        }
        det_net.Forward();

        const float* detResData = detNetResLayer->cpu_data();
        const int nDetections = detNetResLayer->height();
        int goldLabel = atoi(&filename[0]);
        cv::Rect rect;
        for (int i = 0; i < nDetections; i++)
        {
            const int label = detResData[i * 7 + 1];
            if (label != goldLabel
                    && goldLabel != 7)
                continue;

            rect.x = detResData[i * 7 + 3] * img.cols;
            rect.y = detResData[i * 7 + 4] * img.rows;
            rect.width = (detResData[i * 7 + 5] - detResData[i * 7 + 3]) * img.cols;
            rect.height = (detResData[i * 7 + 6] - detResData[i * 7 + 4]) * img.rows;

            if (rect.area() < 20)
                continue;

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
            cv::Mat sample;
            cv::copyMakeBorder(img(rect), sample, top, bottom, left, right, cv::BORDER_CONSTANT, meanPixel);
            cv::imwrite("images/" + filename + ".jpg", sample);

            std::ofstream outFile("annotations/" + filename + ".xml");
            CV_Assert(outFile.is_open());
            outFile << "<annotation>" << std::endl;
            outFile << "  <filename>" << filename + ".jpg" << "</filename>" << std::endl;
            outFile << "  <size>" << std::endl;
            outFile << "    <width>" << sample.cols << "</width>" << std::endl;
            outFile << "    <height>" << sample.rows << "</height>" << std::endl;
            outFile << "    <depth>" << 3 << "</depth>" << std::endl;
            outFile << "  </size>" << std::endl;

            outFile << "  <object>" << std::endl;
            outFile << "    <name>" << filename[0] << "</name>" << std::endl;
            outFile << "    <truncated>" << 0 << "</truncated>" << std::endl;
            outFile << "    <difficult>" << 0 << "</difficult>" << std::endl;
            outFile << "    <bndbox>" << std::endl;
            outFile << "      <xmin>" << 0 << "</xmin>" << std::endl;
            outFile << "      <ymin>" << 0 << "</ymin>" << std::endl;
            outFile << "      <xmax>" << 4 << "</xmax>" << std::endl;
            outFile << "      <ymax>" << 4 << "</ymax>" << std::endl;
            outFile << "    </bndbox>" << std::endl;
            outFile << "  </object>" << std::endl;
            outFile << "</annotation>" << std::endl;

            break;
        }
    }

    return 0;
}
