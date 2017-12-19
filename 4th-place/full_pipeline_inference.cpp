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
cv::Mat rotate(cv::Mat src, double angle)
{
    cv::Mat dst;
    cv::Point2f pt(src.cols/2., src.rows/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
    return dst;
}

class BatchData
{
public:
    BatchData(int size, cv::Size const& dataSize)
        : sequence(std::vector<cv::Mat>(size)),
          dataSize(dataSize)
    {}

    void add(cv::Mat const& img)
    {
        for (size_t i = 1; i < sequence.size(); i++)
        {
            sequence[i - 1] = sequence[i];
        }
        cv::Mat dst;
        cv::resize(img, dst, dataSize);
        sequence[sequence.size() - 1] = dst;
    }

    std::vector<cv::Mat> get() const
    {
        return sequence;
    }

private:
    std::vector<cv::Mat> sequence;
    cv::Size dataSize;
};

struct FrameInfo
{
    int frameId;
    std::vector<float> conf;
    int length;
    std::vector<float> betterConf;
    cv::Rect roi;
};

bool hasFish(std::vector<FrameInfo> const& seqConf)
{
    bool hasFish = false;

    int nWithFish = 0;
    bool shouldTry = false;
    for (size_t i = 0; i < seqConf.size(); i++)
    {
        for (int j = 0; j < 7; j++)
        {
            if (seqConf[i].conf[j] >= 0.6)
                nWithFish++;
            else if (seqConf[i].conf[j] > 0.1)
            {
                shouldTry = true;
            }
        }
    }
    if (nWithFish)
        hasFish = true;
    else if (shouldTry)
    {
        for (size_t i = 0; i < seqConf.size(); i++)
        {
            for (int b = 0; b < 7; b++)
            {
                if (seqConf[i].betterConf[b] >= 0.6)
                    nWithFish++;
            }
        }
    }

    if (nWithFish)
        hasFish = true;

    return hasFish;
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
    caffe::Net<float> seqNet("<path_to>/1_train_sequence_model/deploy5.prototxt", caffe::TEST);
    seqNet.CopyTrainedLayersFrom("<path_to>/sequences.caffemodel");
    caffe::Blob<float>* seqDataLayer = seqNet.input_blobs()[0];
    const int nChns = seqDataLayer->channels();
    const float nSeqCols = seqDataLayer->width();
    const float nSeqRows = seqDataLayer->height();
    const size_t seqBatchSize = seqDataLayer->num();
    cv::Scalar meanPixel(104, 117, 123);
    caffe::Blob<float>* contLayer = seqNet.input_blobs()[1];
    seqNet.Reshape();

    caffe::Net<float> frameClsNet("<path_to>/3_traincls_global_frame_model/deploy.prototxt", caffe::TEST);
    frameClsNet.CopyTrainedLayersFrom("<path_to>/global_cls.caffemodel");
    caffe::Blob<float>* frameClsNetDataLayer = frameClsNet.input_blobs()[0];
    caffe::Blob<float>* frameClsNetResLayer = frameClsNet.output_blobs()[0];

    caffe::Net<float> ssdLocNet("<path_to>/5_train_loc_model/deploy.prototxt", caffe::TEST);
    ssdLocNet.CopyTrainedLayersFrom("<path_to>/localization.caffemodel");
    caffe::Blob<float>* ssdLocDataLayer = ssdLocNet.input_blobs()[0];
    const int nLocChns = ssdLocDataLayer->channels();
    const float nLocCols = ssdLocDataLayer->width();
    const float nLocRows = ssdLocDataLayer->height();
    const size_t batchSize = 1;
    ssdLocDataLayer->Reshape(batchSize, nLocChns, nLocRows, nLocCols);
    ssdLocNet.Reshape();


    caffe::Net<float> clsRoiNet("<path_to>/7_train_cls_roi_model/deploy_roi.prototxt", caffe::TEST);
    clsRoiNet.CopyTrainedLayersFrom("<paht_to>/cls_roi.caffemodel");
    caffe::Blob<float>* clsRoiNetDataLayer = clsRoiNet.input_blobs()[0];
    caffe::Blob<float>* clsRoiNetResLayer = clsRoiNet.output_blobs()[0];

    int delay = 0;
    std::ifstream videoList(params.inputFolder);
    CV_Assert(videoList.is_open());
    std::string filename;
    while (videoList >> filename)
    {
        cv::VideoCapture cap("<path_to>/test_videos/" + filename + ".mp4");
        CV_Assert(cap.isOpened());

        int frameId = -1;
        cv::Mat img;
        cap.read(img);
        BatchData batchData(seqBatchSize, cv::Size(nSeqCols, nSeqRows));
        for (int i = 0; i < seqBatchSize; i++)
        {
            batchData.add(img);
        }
        bool shouldContinue = true;
        int prevFrame = -1;
        int nSeq = 0;
        std::vector<FrameInfo> seqConf;
        std::vector<int> frameIds;
        while (shouldContinue)
        {
            frameId++;
            std::vector<cv::Mat> imgs = batchData.get();
            for (int i = 0; i < seqBatchSize; i++)
            {
                std::vector<cv::Mat> planes(3);
                cv::split(imgs[i], planes);
                for (size_t pId = 0; pId < planes.size(); pId++)
                {
                    cv::Mat dst(nSeqRows, nSeqCols, CV_32FC1, (void*)(seqDataLayer->mutable_cpu_data() + seqDataLayer->offset(i, pId)));
                    planes[pId].convertTo(dst,
                                          CV_32FC1, 1, -meanPixel(pId));
                }
                *(contLayer->mutable_cpu_data() + contLayer->offset(i)) = (i == 0 ? 0 : 1);
            }

            seqNet.Forward();

            boost::shared_ptr<caffe::Blob<float> > detectionsLayer = seqNet.blob_by_name("softmax");
            const float* detectionsData = detectionsLayer->cpu_data();
            bool shouldShow = true;

            for (int i = 0; i < seqBatchSize; i++)
            {
                if (detectionsData[i * 2] >= 0.7)
                {
                    shouldShow = false;
                    break;
                }
            }

            if (shouldShow)
            {
                if (frameId - prevFrame != 1)
                {
                    if (hasFish(seqConf))
                    {
                        nSeq++;
                        for (size_t i = 0; i < seqConf.size(); i++)
                        {
                            std::cout << "0," << frameIds[i] << "," << filename << ","
                                      << nSeq << "," << seqConf[i].length;
                            for (int j = 0; j < 7; j++)
                            {
                                std::cout << "," << seqConf[i].betterConf[j];
                            }
                            std::cout << std::endl;
                        }
                    }
                    seqConf.clear();
                    frameIds.clear();
                }
                prevFrame = frameId;

                int detRows = frameClsNetDataLayer->height();
                int detCols = frameClsNetDataLayer->width();
                cv::Mat resizedImgDet;
                cv::resize(img, resizedImgDet, cv::Size(detCols, detRows));
                std::vector<cv::Mat> planesDet(3);
                cv::split(resizedImgDet, planesDet);
                for (size_t pId = 0; pId < planesDet.size(); pId++)
                {
                    cv::Mat dst(detRows, detCols, CV_32FC1, (void*)(frameClsNetDataLayer->mutable_cpu_data() + frameClsNetDataLayer->offset(0, pId)));
                    planesDet[pId].convertTo(dst,
                                             CV_32FC1, 1, -meanPixel(pId));
                    dst *= 0.017;
                }
                frameClsNet.Forward();

                const float* detResData = frameClsNetResLayer->cpu_data();
                std::vector<float> curConf;
                for (int i = 0; i < 8; i++)
                {
                    curConf.push_back(detResData[i]);
                }
                frameIds.push_back(frameId);
                FrameInfo frameInfo;
                frameInfo.frameId = frameId;
                frameInfo.conf = curConf;

                {
                    cv::Mat resizedImg;
                    cv::resize(img, resizedImg, cv::Size(nLocCols, nLocRows));
                    std::vector<cv::Mat> planesLoc(3);
                    cv::split(resizedImg, planesLoc);
                    for (size_t pId = 0; pId < planesLoc.size(); pId++)
                    {
                        cv::Mat dst(nLocRows, nLocCols, CV_32FC1, (void*)(ssdLocDataLayer->mutable_cpu_data() + ssdLocDataLayer->offset(0, pId)));
                        planesLoc[pId].convertTo(dst,
                                              CV_32FC1, 1, -meanPixel(pId));
                    }
                    ssdLocNet.Forward();

                    boost::shared_ptr<caffe::Blob<float> > detectionsLocLayer = ssdLocNet.blob_by_name("detection_out");
                    const float* detectionsLocData = detectionsLocLayer->cpu_data();
                    const int nLocDetections = detectionsLocLayer->height();
                    cv::Mat sample;
                    cv::Rect rect;
                    int length = 0;
                    for (int i = 0; i < std::min(1, nLocDetections); i++)
                    {
                        const int label = detectionsLocData[i * 7 + 1];
                        if (label == 7)
                            continue;

                        rect.x = detectionsLocData[i * 7 + 3] * img.cols;
                        rect.y = detectionsLocData[i * 7 + 4] * img.rows;
                        rect.width = (detectionsLocData[i * 7 + 5] - detectionsLocData[i * 7 + 3]) * img.cols;
                        rect.height = (detectionsLocData[i * 7 + 6] - detectionsLocData[i * 7 + 4]) * img.rows;

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
                    frameInfo.length = length;
                    frameInfo.betterConf.resize(7, 0);

                    if (!sample.empty())
                    {
                        int clsRoiRows = clsRoiNetDataLayer->height();
                        int clsRoiCols = clsRoiNetDataLayer->width();
                        cv::Mat resizedImgClsRoi;
                        cv::resize(sample, resizedImgClsRoi, cv::Size(clsRoiCols, clsRoiRows));
                        std::vector<cv::Mat> planesClsRoi(3);
                        cv::split(resizedImgClsRoi, planesClsRoi);
                        for (size_t pId = 0; pId < planesClsRoi.size(); pId++)
                        {
                            cv::Mat dst(clsRoiRows, clsRoiCols, CV_32FC1, (void*)(clsRoiNetDataLayer->mutable_cpu_data() + clsRoiNetDataLayer->offset(0, pId)));
                            planesClsRoi[pId].convertTo(dst,
                                                     CV_32FC1, 1, -meanPixel(pId));
                            //dst *= 0.017;
                        }
                        clsRoiNet.Forward();

                        const float* clsResData = clsRoiNetResLayer->cpu_data();

                        float maxConf = -1;
                        int maxPos = 0;
                        for (int i = 0; i < 8; i++)
                        {
                            if (clsResData[i] > maxConf)
                            {
                                maxConf = clsResData[i];
                                maxPos = i;
                            }
                        }

                        {
                            for (size_t i = 0; i < frameInfo.betterConf.size(); i++)
                            {
                                    frameInfo.betterConf[i] = clsResData[i];
                            }
                        }
                    }
                }
                seqConf.push_back(frameInfo);
            }
            shouldContinue = cap.read(img);
            if (shouldContinue)
                batchData.add(img);
        }

        if (hasFish(seqConf))
        {
            nSeq++;
            for (size_t i = 0; i < seqConf.size(); i++)
            {
                std::cout << "0," << frameIds[i] << "," << filename << ","
                          << nSeq << "," << seqConf[i].length;
                for (int j = 0; j < seqConf[i].betterConf.size(); j++)
                {
                    std::cout << "," << seqConf[i].betterConf[j];
                }
                std::cout << std::endl;
            }
        }

        std::cout << filename << " " << nSeq << std::endl;
    }

    return 0;
}
