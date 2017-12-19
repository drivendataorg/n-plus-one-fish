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

    LabeledOject(int id = -1, int frameId = -1, Type type = Empty,
                 cv::Point2f const& pt1 = cv::Point2f(-1, -1),
                 cv::Point2f const& pt2 = cv::Point2f(-1, -1))
        : id(id),
          frameId(frameId),
          type(type),
          pt1(pt1),
          pt2(pt2) {}

    float length() const {return sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y));}

    int id;
    int frameId;
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
    std::string folder(argv[2]);

    std::map<std::string, std::map<int, LabeledOject> > videos;
    std::map<std::string, std::vector<int> > fishCountPerVideo;
    std::string line;
    std::getline(annotations, line); // skip header
    while (std::getline(annotations, line))
    {
        std::vector<std::string> tokens = split(line, ',');
        std::string key = tokens[1];
        if (fishCountPerVideo.find(key) == fishCountPerVideo.end())
        {
            fishCountPerVideo[key] = std::vector<int>(8, 0);
        }

        int frameId = atoi(tokens[2].c_str());
        if (tokens[3] == "")
        {
            videos[key][frameId] = LabeledOject(-1, frameId);
            fishCountPerVideo[key][7]++;
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
        videos[key][frameId] = LabeledOject(id, frameId, (LabeledOject::Type)type, pt1, pt2);
        fishCountPerVideo[key][type]++;
    }
    size_t limits[] = {98, 12, 2000, 10, 2000, 50, 20, 6};
    LabeledOject::Type clsToType[] = {
        LabeledOject::Four, LabeledOject::Grey, LabeledOject::Other,
        LabeledOject::Plaice, LabeledOject::Summer, LabeledOject::Widow,
        LabeledOject::Winter, LabeledOject::Empty};
    for (auto videoIt : videos)
    {
        std::map<int, LabeledOject> species;
        for (int cId = 0; cId < 8; cId++)
        {
            std::vector<LabeledOject> specie;
            for (auto frameIt : videoIt.second)
            {
                if (frameIt.second.type == clsToType[cId])
                {
                    specie.push_back(frameIt.second);
                }
            }
            std::random_shuffle(specie.begin(), specie.end());
            specie.resize(std::min(specie.size(), limits[cId]));
            for (auto s : specie)
            {
                species[s.frameId] = s;
            }
        }
        cv::VideoCapture cap(folder + "/" + videoIt.first + ".mp4");
        CV_Assert(cap.isOpened());
        int frameId = -1;
        cv::Mat img;
        while (cap.read(img))
        {
            frameId++;
            if (species.find(frameId) == species.end())
            {
                continue;
            }
            std::stringstream name;
            name << ((species[frameId].type == -1) ? 7 : species[frameId].type) << "_" << videoIt.first << "_"
                 << std::setw(6) << std::setfill('0') << frameId;
            cv::imwrite("images/" + name.str() + ".jpg", img);

            std::ofstream outFile("annotations/" + name.str() + ".xml");
            CV_Assert(outFile.is_open());
            outFile << "<annotation>" << std::endl;
            outFile << "  <filename>" << name.str() + ".jpg" << "</filename>" << std::endl;
            outFile << "  <size>" << std::endl;
            outFile << "    <width>" << img.cols << "</width>" << std::endl;
            outFile << "    <height>" << img.rows << "</height>" << std::endl;
            outFile << "    <depth>" << 3 << "</depth>" << std::endl;
            outFile << "  </size>" << std::endl;

            outFile << "  <object>" << std::endl;
            outFile << "    <name>" << ((species[frameId].type == -1) ? 7 : species[frameId].type) << "</name>" << std::endl;
            outFile << "    <truncated>" << 0 << "</truncated>" << std::endl;
            outFile << "    <difficult>" << 0 << "</difficult>" << std::endl;
            outFile << "    <bndbox>" << std::endl;
            outFile << "      <xmin>" << 0 << "</xmin>" << std::endl;
            outFile << "      <ymin>" << 0 << "</ymin>" << std::endl;
            outFile << "      <xmax>" << 10 << "</xmax>" << std::endl;
            outFile << "      <ymax>" << 10 << "</ymax>" << std::endl;
            outFile << "    </bndbox>" << std::endl;
            outFile << "  </object>" << std::endl;
            outFile << "</annotation>" << std::endl;
        }
    }

    return 0;
}
