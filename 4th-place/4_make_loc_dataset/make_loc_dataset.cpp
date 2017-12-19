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

void clipRect(cv::Mat const& mask, cv::Rect& roi)
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
    if (roi.x + roi.width > mask.cols)
    {
        int diff = roi.x + roi.width - mask.cols + 1;
        roi.width -= diff;
    }
    if (roi.y + roi.height > mask.rows)
    {
        int diff = roi.y + roi.height - mask.rows + 1;
        roi.height -= diff;
    }
}

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
    std::ifstream videosList(argv[3]);
    CV_Assert(videosList.is_open());
    std::vector<int> classesCount(7, 0);
    std::map<std::string, int> selectedVideos;
    while (videosList >> line)
    {
        std::map<int, LabeledOject> labels = videos[line];

        cv::VideoCapture cap(folder + "/" + line + ".mp4");
        CV_Assert(cap.isOpened());
        int frameId = -1;
        cv::Mat img;
        while (cap.read(img))
        {
            frameId++;
            if (labels.find(frameId) == labels.end())
                continue;
            LabeledOject object = labels[frameId];
            if (object.type == LabeledOject::Empty)
                continue;
            std::stringstream ss;
            ss << object.type << "_" << line << "_" << std::setw(6) << std::setfill('0') << frameId;
            cv::imwrite("images/" + ss.str() + ".jpg", img);

            cv::Rect maxRect;
            {
                cv::Mat res;
                cv::Mat mask(img.size(), CV_8UC1);
                mask.setTo(cv::GC_BGD);
                cv::Point ptCenter;
                ptCenter.x = std::min(object.pt1.x, object.pt2.x) + abs(object.pt1.x - object.pt2.x) / 2;
                ptCenter.y = std::min(object.pt1.y, object.pt2.y) + abs(object.pt1.y - object.pt2.y) / 2;
                int side = object.length() / 2;
                cv::Rect prBg(ptCenter.x - side, ptCenter.y - side, side * 2, side * 2);
                clipRect(mask, prBg);
                mask(prBg).setTo(cv::GC_PR_BGD);
                side = side * 0.1;
                //cv::Rect prFg(ptCenter.x - side, ptCenter.y - side, side * 2, side * 2);
                //mask(prFg).setTo(cv::GC_PR_FGD);
                cv::line(mask, object.pt1, object.pt2, cv::GC_FGD, 2);
                cv::Mat bgd, fgd;
                cv::grabCut(img, mask, cv::Rect(), bgd, fgd, 2, cv::GC_INIT_WITH_MASK);
                cv::Mat segmented = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
                img.copyTo(res, segmented);
                //cv::rectangle(frame, prFg, cv::Scalar(255, 0, 255), 1);

                segmented *= 255;
                std::vector<std::vector<cv::Point> > contours;
                std::vector<cv::Vec4i> hierarchy;
                cv::findContours(segmented, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

                for (size_t i = 0; i < contours.size(); i++)
                {
                    if (contours[i].size() < 5)
                        continue;
                    cv::Rect curRect = cv::boundingRect(contours[i]);
                    if (curRect.area() > maxRect.area())
                    {
                        maxRect = curRect;
                    }
                }
            }


            std::ofstream outFile("annotations/" + ss.str() + ".xml");
            CV_Assert(outFile.is_open());
            outFile << "<annotation>" << std::endl;
            outFile << "  <filename>" << ss.str() + ".jpg" << "</filename>" << std::endl;
            outFile << "  <size>" << std::endl;
            outFile << "    <width>" << img.cols << "</width>" << std::endl;
            outFile << "    <height>" << img.rows << "</height>" << std::endl;
            outFile << "    <depth>" << 3 << "</depth>" << std::endl;
            outFile << "  </size>" << std::endl;

            outFile << "  <object>" << std::endl;
            outFile << "    <name>" << object.type << "</name>" << std::endl;
            outFile << "    <truncated>" << 0 << "</truncated>" << std::endl;
            outFile << "    <difficult>" << 0 << "</difficult>" << std::endl;
            outFile << "    <bndbox>" << std::endl;
            outFile << "      <xmin>" << maxRect.x << "</xmin>" << std::endl;
            outFile << "      <ymin>" << maxRect.y << "</ymin>" << std::endl;
            outFile << "      <xmax>" << maxRect.x + maxRect.width << "</xmax>" << std::endl;
            outFile << "      <ymax>" << maxRect.y + maxRect.height << "</ymax>" << std::endl;
            outFile << "    </bndbox>" << std::endl;
            outFile << "  </object>" << std::endl;
            outFile << "</annotation>" << std::endl;

        }
    }

    return 0;
}
