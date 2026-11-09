#include "armor_detector/detector_nodelet.h"
#include <pluginlib/class_list_macros.h>
#include <algorithm>

namespace armor_detector
{

    void DetectorNodelet::onInit()
    {
        setlocale(LC_ALL, "");

        ros::NodeHandle n = getPrivateNodeHandle();

        std::string input_topic = "/ft/debug/image";
        std::string debug_topic = "/ft/detector/image";
        int queue_size = 1;

        n.param<std::string>("input_topic", input_topic, input_topic);
        n.param<std::string>("debug_topic", debug_topic, debug_topic);
        n.param<int>("queue_size", queue_size, queue_size);

        n.param<int>("BH_MIN", BH_MIN_, BH_MIN_);
        n.param<int>("BH_MAX", BH_MAX_, BH_MAX_);
        n.param<int>("RH_MIN", RH_MIN_, RH_MIN_);
        n.param<int>("RH_MAX", RH_MAX_, RH_MAX_);

        n.param<int>("RH2_MIN", RH2_MIN_, RH2_MIN_);
        n.param<int>("RH2_MAX", RH2_MAX_, RH2_MAX_);

        n.param<int>("S_MIN", S_MIN_, S_MIN_);
        n.param<int>("S_MAX", S_MAX_, S_MAX_);
        n.param<int>("V_MIN", V_MIN_, V_MIN_);
        n.param<int>("V_MAX", V_MAX_, V_MAX_);
        n.param<int>("min_area", MIN_AREA_, MIN_AREA_);

        kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

        debug_pub_ = n.advertise<sensor_msgs::Image>(debug_topic, 1);
        image_sub_ = n.subscribe(input_topic, queue_size, &DetectorNodelet::imageCallback, this);


        NODELET_INFO("detector_nodelet: subscribe %s -> publish %s (red+blue)",input_topic.c_str(), debug_topic.c_str());
    }

    void DetectorNodelet::getcontours(const cv::Mat &g_mask, cv::Mat &g_img,bool is_red_armor)
    {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::RotatedRect> bars;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(g_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < (int)contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if (area <= MIN_AREA_)
                continue;

            cv::RotatedRect rr = cv::minAreaRect(contours[i]);
            float ang = rr.angle;
            if (rr.size.width > rr.size.height)
            {
                std::swap(rr.size.width, rr.size.height);
                ang += 90.0f;
            }
            while (ang >= 90.0f)
                ang -= 180.0f;
            while (ang < -90.0f)
                ang += 180.0f;
            rr.angle = ang;

            bars.push_back(rr);
        }

        struct Pair
        {
            int i, j;
            float cost;
        };
        std::vector<Pair> cands;

        const float GAP_RATIO_MIN = 1.5f;
        const float GAP_RATIO_MAX = 5.0f;
        const float GAP_RATIO_IDEAL = (GAP_RATIO_MIN + GAP_RATIO_MAX) * 0.5f;

        for (int i = 0; i < (int)bars.size(); i++)
        {
            for (int j = i + 1; j < (int)bars.size(); j++)
            {
                // 高度差
                float dy = std::abs(bars[i].center.y - bars[j].center.y);
                float dx = std::abs(bars[i].center.x - bars[j].center.x);
                float maxLen = std::max(bars[i].size.height, bars[j].size.height);
                if (maxLen <= 0)
                    continue;
                if (dy > 0.4f * maxLen)
                    continue;

                // 面积差
                float a1 = bars[i].size.area();
                float a2 = bars[j].size.area();
                float aMax = std::max(a1, a2);
                float aMin = std::min(a1, a2);
                if (aMax - aMin > aMax * 0.5f)
                    continue;

                // 角度差（处理跨 ±90）
                float dAng = std::abs(bars[i].angle - bars[j].angle);
                if (dAng > 90.0f)
                    dAng = 180.0f - dAng;
                if (dAng > 10.0f)
                    continue;

                // 间距
                float avgLen = (bars[i].size.height + bars[j].size.height) * 0.5f;
                if (avgLen <= 0)
                    continue;
                float ratio = dx / avgLen;
                if (ratio < GAP_RATIO_MIN || ratio > GAP_RATIO_MAX)
                    continue;

                float devRatio = std::abs(ratio - GAP_RATIO_IDEAL);

                float cost = dAng * 2.0f + (dy / maxLen) * 20.0f + (aMax - aMin) / aMax * 20.0f + devRatio * 5.0f;

                Pair p;
                p.i = i;
                p.j = j;
                p.cost = cost;
                cands.push_back(p);
            }
        }

        std::sort(cands.begin(), cands.end(),
                  [](const Pair &a, const Pair &b)
                  { return a.cost < b.cost; });

        std::vector<bool> used(bars.size(), false);

        cv::Scalar center_color = is_red_armor ? cv::Scalar(0, 0, 255): cv::Scalar(0, 255, 0);

        for (size_t k = 0; k < cands.size(); k++)
        {
            const Pair &c = cands[k];
            if (used[c.i] || used[c.j])
                continue;
            used[c.i] = true;
            used[c.j] = true;

            // 画灯条框（绿色）
            cv::Point2f pi[4], pj[4];
            bars[c.i].points(pi);
            bars[c.j].points(pj);
            for (int m = 0; m < 4; m++)
            {
                cv::line(g_img, pi[m], pi[(m + 1) % 4], cv::Scalar(0, 255, 0), 2);
                cv::line(g_img, pj[m], pj[(m + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }

            // 画装甲板中心（红色圆点）
            cv::Point2f armorCenter = (bars[c.i].center + bars[c.j].center) * 0.5f;
            cv::circle(g_img, armorCenter, 5, center_color, -1);
        }
    }

    void DetectorNodelet::update(cv::Mat &g_img, cv::Mat &mask)
    {
        cv::Mat g_hsv;
        cv::cvtColor(g_img, g_hsv, cv::COLOR_BGR2HSV);

        cv::Mat eroded;

        // ================= 蓝色 → 友方，绿色中心 =================
        cv::inRange(g_hsv, cv::Scalar(BH_MIN_, S_MIN_, V_MIN_),
                    cv::Scalar(BH_MAX_, S_MAX_, V_MAX_), mask);
        cv::erode(mask, eroded, kernel_);
        cv::dilate(eroded, mask, kernel_);
        getcontours(mask, g_img, false);

        // ================= 红色 → 敌方，红色中心 =================
        cv::Mat red_low, red_high;
        cv::inRange(g_hsv, cv::Scalar(RH_MIN_, S_MIN_, V_MIN_),
                    cv::Scalar(RH_MAX_, S_MAX_, V_MAX_), red_low);
        cv::inRange(g_hsv, cv::Scalar(RH2_MIN_, S_MIN_, V_MIN_),
                    cv::Scalar(RH2_MAX_, S_MAX_, V_MAX_), red_high);
        cv::bitwise_or(red_low, red_high, mask);
        cv::erode(mask, eroded, kernel_);
        cv::dilate(eroded, mask, kernel_);
        getcontours(mask, g_img, true);
    }

    void DetectorNodelet::imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        cv_bridge::CvImagePtr cv_ptr =
            cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        cv::Mat img = cv_ptr->image;
        cv::Mat mask;
        update(img, mask);

        sensor_msgs::ImagePtr out =
            cv_bridge::CvImage(msg->header, "bgr8", img).toImageMsg();
        debug_pub_.publish(*out);
    }

} // namespace armor_detector

PLUGINLIB_EXPORT_CLASS(armor_detector::DetectorNodelet, nodelet::Nodelet)