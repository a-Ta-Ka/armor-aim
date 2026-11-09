

// cd /root/vision_ws/scripts && ./hsv_tune

// cd / root / vision_ws / scripts g++ hsv_tune.cpp - o hsv_tune `pkg - config-- cflags-- libs opencv4`


#include <opencv2/opencv.hpp>
#include <cstdio>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

static Mat g_img, g_hsv, g_mask,g_a, maskSmall;
static std::vector<std::string> g_files;

static int g_idx = 0;
static int BH_MIN = 80, BH_MAX = 120,
           RH_MIN = 0, RH_MAX = 40;
static int S_MIN = 80, S_MAX = 255;
static int V_MIN = 160, V_MAX = 255;

static cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
static bool red = false;
static const char *WIN = "image";

static void getcontours(Mat g_mask,Mat g_img)
{
    std::vector<std::vector<cv::Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(g_mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size();i++)
    {
        int area = contourArea(contours[i]);

        if(area<=10)
        {
            continue;
        }

        vector<vector<Point>> conpoly(contours.size());
        vector<Rect> boundRect(contours.size());

        double peri = arcLength(contours[i], true);
        approxPolyDP(contours[i], conpoly[i], 0.02 * peri, true);
        boundRect[i] = boundingRect(conpoly[i]);
        if ((float)boundRect[i].width / (float)(boundRect[i].height) >= 0.5 )
        {
            continue;
        }
            rectangle(g_img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2);
    }
}

static void update()
{
    if(red)
    {
        cv::inRange(g_hsv, cv::Scalar(RH_MIN, S_MIN, V_MIN),
        cv::Scalar(RH_MAX, S_MAX, V_MAX), g_mask);
    }else
    {
        cv::inRange(g_hsv, cv::Scalar(BH_MIN, S_MIN, V_MIN),
        cv::Scalar(BH_MAX, S_MAX, V_MAX), g_mask);
    }
    cv::erode(g_mask, g_a, kernel);
    cv::dilate(g_a, g_mask, kernel);
    getcontours(g_mask,g_img);
}

static void load(int idx)
{
    g_idx = (idx + (int)g_files.size()) % (int)g_files.size();
    g_img = cv::imread(g_files[g_idx], cv::IMREAD_COLOR);
    if (g_img.empty())
        return;
    cv::cvtColor(g_img, g_hsv, cv::COLOR_BGR2HSV);
    update();
}

int main()
{
    for (int i = 0; i < 20; i++)
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "/root/vision_ws/frames/frame_%03d.jpg", i);
        g_files.push_back(buf);
    }

    load(0);
    while (true)
    {
        cv::imshow("image", g_img);
        cv::imshow("mask", g_mask);
        int k = cv::waitKey(30);
        if (k == 'r')
        {
            red = !red;
            update();
        }
        if (k == 27 || k == 'q')
            break;
        if (k == 'n')
            load(g_idx + 1);
        if (k == 'p')
            load(g_idx - 1);
    }
    return 0;
}
