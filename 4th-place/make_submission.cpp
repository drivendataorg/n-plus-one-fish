#include <opencv2/opencv.hpp>

#include <fstream>

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

struct BestConf
{
    int pos;
    float val;
};

struct Detection
{
    int id;
    int frameId;
    int length;
    int getConf(float& val)
    {
        int maxPos = 0;
        val = conf[maxPos];
        for (size_t j = 1; j < conf.size(); j++)
        {
            if (conf[j] > val)
            {
                maxPos = j;
                val = conf[j];
            }
        }
        return maxPos;
    }
    std::vector<float> conf;
    std::vector<std::string> tokens;
};

int main(int argc, char* argv[])
{
    std::ifstream sampleList(argv[1]);
    CV_Assert(sampleList.is_open());
    std::map<std::string, std::vector<std::string> > sampleSubmission;
    std::vector<std::string> keysOrdered;
    std::string header;
    std::getline(sampleList, header);

    std::string line;

    while (std::getline(sampleList, line))
    {
        std::vector<std::string> tokens = split(line, ',');
        std::string key = tokens[1] + "," + tokens[2];
        keysOrdered.push_back(key);
        tokens[3] = "-nan";
        sampleSubmission[key] = tokens;
    }

    size_t lineId = 0;
    size_t maxLines = 4591730;
    std::map<std::string, std::map<int, std::vector<Detection> > > detById; 
    std::ifstream result(argv[2]);
    CV_Assert(result.is_open());

    lineId = 0;
    while (std::getline(result, line))
    {
        if (lineId++ > maxLines)
            break;
        std::vector<std::string> tokens = split(line, ',');        
        std::string key = tokens[1] + "," + tokens[2];
        if (sampleSubmission.find(key) != sampleSubmission.end())
        {
            for (int j = 3; j < tokens.size(); j++)
            {
                sampleSubmission[key][j] = tokens[j];
            }
        }

        Detection detection;
        detection.id = atof(tokens[3].c_str());
        detection.frameId = atof(tokens[1].c_str());        
        detection.length = atof(tokens[4].c_str());
        for (size_t i = 5; i < tokens.size(); i++)
        {
            detection.conf.push_back(atof(tokens[i].c_str()));
        }
        detection.tokens = tokens;
        detById[tokens[2]][detection.id].push_back(detection);
    }
    std::cout << lineId << std::endl;

    std::ofstream outF("result3.csv");
    outF << header << std::endl;
    for (auto const& key : keysOrdered)
    {
        std::vector<std::string> res = sampleSubmission[key];
        outF << res[0];
        for (int i = 1; i < res.size(); i++)
        {
            outF << "," << res[i];
        }
        outF << std::endl;
    }

    return 0;
}
