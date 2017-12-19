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

    std::map<std::string, std::map<int, std::vector<Detection> > > videos;
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

        videos[key][detection.id].push_back(detection);
    }
    for (auto& videoIt : videos)
    {
        std::vector<int> leftIds;
        for (auto& fishIt : videoIt.second)
        {
            std::vector<Detection> filteredDetections;
            for (size_t i = 0; i < fishIt.second.size(); i++)
            {
                float confPerDetection = 0;
                for (int j = 0; j < fishIt.second[i].conf.size(); j++)
                {
                    confPerDetection += fishIt.second[i].conf[j];
                }
                if (confPerDetection < 0.5)
                    continue;
                filteredDetections.push_back(fishIt.second[i]);
            }
            fishIt.second = filteredDetections;
            if (!filteredDetections.empty())
                leftIds.push_back(fishIt.first);
        }
        std::map<int, int> idMap;
        for (size_t i = 0; i < leftIds.size(); i++)
        {
            idMap[leftIds[i]] = i + 1;
        }

        for (auto& fishIt : videoIt.second)
        {
            if (fishIt.second.empty())
                continue;

            std::vector<std::vector<Detection> > instanceSeq;
            int prevCId = -2;
            for (size_t i = 0; i < fishIt.second.size(); i++)
            {
                float conf = 0;
                int cId = getConf(conf, fishIt.second[i].conf);
                if (cId != prevCId)
                {
                    instanceSeq.push_back(std::vector<Detection>());
                }
                prevCId = cId;
                instanceSeq.back().push_back(fishIt.second[i]);
            }

            int idMaxSeq = 0;
            float maxConf = 0;
            for (size_t sId = 0; sId < instanceSeq.size(); sId++)
            {
                float maxSeqConf = 0;
                for (size_t i = 0; i < instanceSeq[sId].size(); i++)
                {
                    float conf;
                    int clsId = getConf(conf, instanceSeq[sId][i].conf);
                    if (conf > maxSeqConf)
                    {
                        maxSeqConf = conf; // was
                    }
                }
                if (maxSeqConf > maxConf)
                {
                    maxConf = maxSeqConf;
                    idMaxSeq = sId;
                }
            }
            fishIt.second = instanceSeq[idMaxSeq];
        }
        for (auto fishIt : videoIt.second)
        {
            for (auto detection : fishIt.second)
            {
                std::cout << "0," << detection.frameId << "," << videoIt.first << "," << idMap[fishIt.first] << ","
                          << detection.length;
                for (size_t i = 0; i < detection.conf.size(); i++)
                {
                    std::cout << "," << detection.conf[i];
                }
                std::cout << std::endl;
            }
        }
    }

    return 0;
}
