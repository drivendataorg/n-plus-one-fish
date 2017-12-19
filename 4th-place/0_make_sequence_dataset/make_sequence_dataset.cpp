#include <opencv2/opencv.hpp>

#include <fstream>
#include <iomanip>

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

int main(int argc, char* argv[])
{
    std::ifstream sequences(argv[1]);
    CV_Assert(sequences.is_open());

    std::string folder(argv[2]);
    int nMax = 10000000;
    std::string line;
    int id = 0;
    std::ofstream cont("continuations.txt");
    std::ofstream labels("labels.txt");
    while (std::getline(sequences, line))
    {
        std::vector<std::string> tokens = split(line, ',');
        cv::VideoCapture cap(folder + "/" + tokens[0] + ".mp4");
        CV_Assert(cap.isOpened());
        int label = atoi(tokens[1].c_str());
        std::vector<int> frameIds;
        for (int i = 2; i < 2 + 11; i++)
        {
            frameIds.push_back(atoi(tokens[i].c_str()));
        }
        int frameId = 0;
        cv::Mat frame;
        cap.read(frame);
        for (size_t i = 0; i < frameIds.size(); i++)
        {
            while (frameId != frameIds[i])
            {
                cap.read(frame);
                frameId++;
            }
            std::stringstream ss;
            ss << std::setw(6) << std::setfill('0') << id << "_" << tokens[0] << "_"
               << std::setw(6) << std::setfill('0') << frameId << "_"
               << std::setw(3) << std::setfill('0') << i << ".jpg";
            cv::imwrite(ss.str(), frame);
            cont << (i == 0 ? 0 : 1) << std::endl;
            labels << label << std::endl;
        }

        id++;
        if (id % 100 == 0)
        {
            std::cout << id << std::endl;
        }
        if (id >= nMax)
            break;
    }

    return 0;
}
