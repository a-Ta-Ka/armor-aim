#include "armor_detector/camera_nodelet.h"
#include <pluginlib/class_list_macros.h>

namespace armor_detector
{

    void CameraNodelet::onInit()
    {
        setlocale(LC_ALL, "");

        // 原来 main 里的 n，现在由 manager 提供
        ros::NodeHandle n = getPrivateNodeHandle();
        image_transport::ImageTransport it(n);

        std::string image_topic;
        std::string camera_info_topic;
        std::string debug_topic;
        int queue_size = 1;

        n.param<std::string>("image_topic", image_topic, "/hk_camera/image_raw");
        n.param<std::string>("camera_info_topic", camera_info_topic, "/hk_camera/camera_info");
        n.param<std::string>("debug_topic", debug_topic, "/ft/debug/image");
        n.param<int>("queue_size", queue_size, 1);

        debug_pub_ = it.advertise(debug_topic, queue_size);

        image_transport::TransportHints hints("compressed", ros::TransportHints(), n);

        image_sub_ = it.subscribe(image_topic, queue_size,
                                  &CameraNodelet::imageCallback, this, hints);

        info_sub_ = n.subscribe(camera_info_topic, queue_size,
                                &CameraNodelet::cameraInfoCallback, this);

        NODELET_INFO("camera_nodelet started: %s -> %s",
                     image_topic.c_str(), debug_topic.c_str());
    }

    void CameraNodelet::imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
        const cv::Mat &image = cv_ptr->image;

        sensor_msgs::ImagePtr out =
            cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();
        debug_pub_.publish(out);
    }

    void CameraNodelet::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr &msg)
    {
        if (msg->K[0] == 0.0)
        {
            NODELET_WARN("K是空的");
            return;
        }

        K_ = cv::Mat(3, 3, CV_64F, (void *)msg->K.data()).clone();

        D_.release();
        if (!msg->D.empty())
        {
            D_ = cv::Mat(msg->D.size(), 1, CV_64F, (void *)msg->D.data()).clone();
        }

        img_width_ = msg->width;
        img_height_ = msg->height;

        NODELET_INFO("像素是%dx%d, 畸变模型=%s",
                     img_width_, img_height_, msg->distortion_model.c_str());
    }

} // namespace armor_detector

// 注册插件：armor_detector/CameraNodelet
PLUGINLIB_EXPORT_CLASS(armor_detector::CameraNodelet, nodelet::Nodelet)