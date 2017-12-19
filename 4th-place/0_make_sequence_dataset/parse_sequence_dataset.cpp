#include <opencv2/opencv.hpp>

#include <fstream>
#include <map>

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

struct LabeledOject
{
    enum Type {Empty = -1, Four = 0, Grey, Other, Plaice, Summer, Widow, Winter};

    LabeledOject(int id = -1, Type type = Empty,
                 cv::Point2f const& pt1 = cv::Point2f(-1, -1),
                 cv::Point2f const& pt2 = cv::Point2f(-1, -1))
        : id(id),
          type(type),
          pt1(pt1),
          pt2(pt2) {}

    float length() const {return sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y));}

    int id;
    Type type;
    cv::Point2f pt1, pt2;
};

struct Entry
{
    std::vector<int> frameId;
    int label;
};

int main(int argc, char* argv[])
{
    std::ifstream annotations(argv[1]);
    CV_Assert(annotations.is_open());

    std::map<std::string, std::map<int, LabeledOject> > videos;
    std::string line;
    std::getline(annotations, line); // skip header
    while (std::getline(annotations, line))
    {
        std::vector<std::string> tokens = split(line, ',');
        std::string key = tokens[1];
        int frameId = atoi(tokens[2].c_str());
        if (tokens[3] == "")
        {
            videos[key][frameId] = LabeledOject();
            continue;
        }

        int id = atof(tokens[3].c_str());
        cv::Point2f pt1, pt2;
        pt1.x = atof(tokens[5].c_str());
        pt1.y = atof(tokens[6].c_str());
        pt2.x = atof(tokens[7].c_str());
        pt2.y = atof(tokens[8].c_str());
        int type = -1;
        for (int i = 9; i < 9 + 7; i++)
        {
            if (tokens[i] != "0")
            {
                type = i - 9;
                break;
            }
        }
        CV_Assert(type != -1);
        videos[key][frameId] = LabeledOject(id, (LabeledOject::Type)type, pt1, pt2);
    }
    std::map<std::string, std::vector<Entry> > batches;

    std::string folder(argv[2]);
    int totalMinDist = 100000;
    for (auto videoIt : videos)
    {
        cv::VideoCapture cap(folder + "/" + videoIt.first + ".mp4");
        CV_Assert(cap.isOpened());
        int length = cap.get(CV_CAP_PROP_FRAME_COUNT);

        std::vector<int> centerFrameIds;
        for (auto const& frameIt : videoIt.second)
        {
            centerFrameIds.push_back(frameIt.first);
        }
        std::sort(centerFrameIds.begin(), centerFrameIds.end());
        int nFrames = 5;
        for (size_t i = 0; i < centerFrameIds.size(); i++)
        {
            Entry entry;
            for (int j = centerFrameIds[i] - nFrames; j < centerFrameIds[i] + nFrames + 1; j++)
            {
                if (j < 0)
                    entry.frameId.push_back(0);
                else if (j >= length)
                    entry.frameId.push_back(length - 1);
                else
                    entry.frameId.push_back(j);
            }
            entry.label = 0;
            std::map<int, LabeledOject> labels = videoIt.second;
            if (labels[centerFrameIds[i]].type != LabeledOject::Empty)
            {
                entry.label = 1;
            }
            batches[videoIt.first].push_back(entry);
        }
    }
    std::vector<std::string> records;
    for (auto batch : batches)
    {
        std::random_shuffle(batch.second.begin(), batch.second.end());
        std::sort(batch.second.begin(), batch.second.end(), [=](Entry const& l, Entry const& r)
        {
            return l.label < r.label;
        });

        int nNeg = 0;
        while (batch.second[nNeg].label == 0)
        {
            std::stringstream record;
            record << batch.first << "," << batch.second[nNeg].label << ",";
            for (size_t i = 0; i < batch.second[nNeg].frameId.size(); i++)
            {
                record << batch.second[nNeg].frameId[i] << ",";
            }
            records.push_back(record.str());
            nNeg++;
        }
        for (int j = nNeg; j < std::min(nNeg * 2, (int)batch.second.size()); j++)
        {

            std::stringstream record;
            record << batch.first << "," << batch.second[j].label << ",";
            for (size_t i = 0; i < batch.second[j].frameId.size(); i++)
            {
                record << batch.second[j].frameId[i] << ",";
            }
            records.push_back(record.str());
        }
    }
    std::random_shuffle(records.begin(), records.end());
    std::ofstream outF("sequences_11.txt");
    CV_Assert(outF.is_open());
    for (auto const& record : records)
    {
        outF << record << std::endl;
    }


    std::cout << totalMinDist << std::endl;

    return 0;
}
