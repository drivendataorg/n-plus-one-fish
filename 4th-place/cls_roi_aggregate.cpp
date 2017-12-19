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

struct Detection
{
    int id;
    int frameId;
    int length;
    std::vector<float> conf;
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

int getConf(float& val, std::vector<float> const& conf)
{
    int maxPos = 0;
    val = conf[maxPos];
    for (size_t i = 1; i < conf.size(); i++)
    {
        if (conf[i] >= val) // > ???
        {
            maxPos = i;
            val = conf[i];
        }
    }

    return maxPos;
}

int main (int argc, char* argv[])
{
    std::ifstream annotations(argv[1]);
    CV_Assert(annotations.is_open());

    std::map<std::string, std::map<int, Detection> > videos;
    std::string line;
    while (std::getline(annotations, line))
    {
        std::vector<std::string> tokens = split(line, ',');
        std::string key = tokens[2];
        Detection detection;
        detection.frameId = atoi(tokens[1].c_str());
        detection.id = atof(tokens[3].c_str());
        detection.length = atof(tokens[4].c_str());
        for (int i = 0; i < 7; i++)
        {
            detection.conf.push_back(atof(tokens[5 + i].c_str()));
        }

        videos[key][detection.frameId] = detection;
    }

    for (int fId = 2; fId < argc; fId++)
    {
        std::ifstream nextAnnotations(argv[fId]);
        CV_Assert(nextAnnotations.is_open());

        while (std::getline(nextAnnotations, line))
        {
            std::vector<std::string> tokens = split(line, ',');
            std::string key = tokens[2];
            Detection detection;
            detection.frameId = atoi(tokens[1].c_str());
            detection.id = atof(tokens[3].c_str());
            detection.length = atof(tokens[4].c_str());
            for (int i = 0; i < 7; i++)
            {
                detection.conf.push_back(atof(tokens[5 + i].c_str()));
            }
            if (videos[key].find(detection.frameId) == videos[key].end())
            {
                videos[key][detection.frameId] = detection;
            } else
            {
                for (size_t i = 0; i < 7; i++)
                {
                    videos[key][detection.frameId].conf[i] += detection.conf[i];
                }
            }
        }
    }

    for (auto videoIt : videos)
    {
        std::vector<Detection> detections;
        for (auto frameIt : videoIt.second)
        {
            detections.push_back(frameIt.second);
        }
        std::sort(detections.begin(), detections.end(), [=](Detection const& l, Detection const& r)
        {
           return l.frameId < r.frameId;
        });
        for (auto const& detection : detections)
        {
            std::cout << "0," << detection.frameId << "," << videoIt.first << "," << detection.id << ","
                      << detection.length;

            for (size_t i = 0; i < detection.conf.size(); i++)
            {
                std::cout << "," << detection.conf[i];
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
