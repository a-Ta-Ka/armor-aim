#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <string>
#include <vector>

using namespace std;
using namespace cv;

static int BH_MIN = 80, BH_MAX = 120,
           RH_MIN = 0, RH_MAX = 40;
static int S_MIN = 80, S_MAX = 255;
static int V_MIN = 160, V_MAX = 255;
static int MIN_AREA = 10;

static double MIN_H_W_RATIO = 2.0;

static cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
static bool red = false;

static ros::Publisher debug_pub;

static void getcontours(const Mat &g_mask, Mat &g_img)
{
    vector<vector<Point>> contours;
    vector<RotatedRect> bars;
    vector<Vec4i> hierarchy;
    findContours(g_mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < (int)contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area <= MIN_AREA)
            continue;

        RotatedRect rr = minAreaRect(contours[i]);
        float ang = rr.angle;
        if (rr.size.width > rr.size.height)
        {
            std::swap(rr.size.width, rr.size.height);
            ang += 90.0f;
        }
        while (ang >= 90.0f) ang -= 180.0f;
        while (ang < -90.0f) ang += 180.0f;
        rr.angle = ang;

        bars.push_back(rr);
    }
    struct Pair
    {
        int i, j;
        float cost;
    };
    vector<Pair> cands;
    for (int i = 0; i < bars.size(); i++)
    {
        for (int j = i + 1; j < bars.size(); j++)
        {
            // 高度差
            float dy = std::abs(bars[i].center.y - bars[j].center.y);
            float dx = std::abs(bars[i].center.x - bars[j].center.x);
            float maxLen = std::max(bars[i].size.height, bars[j].size.height);
            if (maxLen <= 0)
                continue;
            if (dy > 0.3f * maxLen)
                continue;

            //面积差
            float a1 = bars[i].size.area();
            float a2 = bars[j].size.area();
            float aMax = std::max(a1, a2);
            float aMin = std::min(a1, a2);
            if (aMax - aMin > aMax * 0.4)
                continue;

            // 角度差（处理跨 ±90）
            float dAng = std::abs(bars[i].angle - bars[j].angle);
            if (dAng > 90.0f)
                dAng = 180.0f - dAng;
            if (dAng > 10.0f)
                continue;

            //间距
            float avgLen = (bars[i].size.height + bars[j].size.height) * 0.5f;
            if (avgLen <= 0)
                continue;
            float ratio = dx / avgLen;
            bool inRangeA = (ratio >= 1.0f && ratio <= 3.0f);
            bool inRangeB = (ratio >= 3.0f && ratio <= 6.5f);
            if (!inRangeA && !inRangeB)
                continue;

            float idealA = 2.0f, idealB = 4.0f;
            float devA = std::abs(ratio - idealA);
            float devB = std::abs(ratio - idealB);
            float devRatio = std::min(devA, devB);

            float cost = dAng * 2.0f + (dy / maxLen) * 20.0f + (aMax - aMin) / aMax * 20.0f + devRatio * 5.0f;

            Pair p;
            p.i = i;
            p.j = j;
            p.cost = cost;
            cands.push_back(p);
        }
    }

    sort(cands.begin(), cands.end(),[](const Pair &a, const Pair &b){ return a.cost < b.cost; });

    vector<bool> used(bars.size(), false);

    for (int k = 0; k < cands.size(); k++)
    {
        const Pair &c = cands[k];
        if (used[c.i] || used[c.j])
            continue;
        used[c.i] = true;
        used[c.j] = true;

        // ---- 画灯条框（绿色） ----
        cv::Point2f pi[4], pj[4];
        bars[c.i].points(pi);
        bars[c.j].points(pj);
        for (int m = 0; m < 4; m++)
        {
            cv::line(g_img, pi[m], pi[(m + 1) % 4], Scalar(0, 255, 0), 2);
            cv::line(g_img, pj[m], pj[(m + 1) % 4], Scalar(0, 255, 0), 2);
        }

        // ---- 画装甲板中心（红色圆点） ----
        cv::Point2f armorCenter = (bars[c.i].center + bars[c.j].center) * 0.5f;
        cv::circle(g_img, armorCenter, 4, Scalar(0, 0, 255), -1);
    }
}

static void update(cv::Mat &g_img, cv::Mat &mask)
{
    Mat g_hsv;
    cvtColor(g_img, g_hsv, cv::COLOR_BGR2HSV);

    if (red)
    {
        cv::inRange(g_hsv, cv::Scalar(RH_MIN, S_MIN, V_MIN),
                    cv::Scalar(RH_MAX, S_MAX, V_MAX), mask);
    }
    else
    {
        cv::inRange(g_hsv, cv::Scalar(BH_MIN, S_MIN, V_MIN),
                    cv::Scalar(BH_MAX, S_MAX, V_MAX), mask);
    }

    Mat a;
    cv::erode(mask, a, kernel);
    cv::dilate(a, mask, kernel);

    getcontours(mask, g_img);
}

static void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    Mat img = cv_ptr->image;
    Mat mask;
    update(img, mask);

    sensor_msgs::ImagePtr out =
        cv_bridge::CvImage(msg->header, "bgr8", img).toImageMsg();
    debug_pub.publish(*out);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "detector_node");
    ros::NodeHandle n("~");

    string input_topic = "/ft/debug/image";
    string debug_topic = "/ft/detector/image";
    int queue_size = 1;
    bool is_red = false;

    n.param<string>("input_topic", input_topic, input_topic);
    n.param<string>("debug_topic", debug_topic, debug_topic);
    n.param<int>("queue_size", queue_size, queue_size);
    n.param<bool>("is_red", is_red, is_red);

    n.param<int>("BH_MIN", BH_MIN, BH_MIN);
    n.param<int>("BH_MAX", BH_MAX, BH_MAX);
    n.param<int>("RH_MIN", RH_MIN, RH_MIN);
    n.param<int>("RH_MAX", RH_MAX, RH_MAX);
    n.param<int>("S_MIN", S_MIN, S_MIN);
    n.param<int>("S_MAX", S_MAX, S_MAX);
    n.param<int>("V_MIN", V_MIN, V_MIN);
    n.param<int>("V_MAX", V_MAX, V_MAX);
    n.param<int>("min_area", MIN_AREA, MIN_AREA);
    n.param<double>("min_h_w_ratio", MIN_H_W_RATIO, MIN_H_W_RATIO);

    red = is_red;

    debug_pub = n.advertise<sensor_msgs::Image>(debug_topic, 1);

    ros::Subscriber sub = n.subscribe(input_topic, queue_size, imageCallback);

    ROS_INFO("detector_node: subscribe %s -> publish %s (red=%d, h/w>=%.2f)",
             input_topic.c_str(), debug_topic.c_str(), (int)red, MIN_H_W_RATIO);

    ros::spin();
    return 0;
}
