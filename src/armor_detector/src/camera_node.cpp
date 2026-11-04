#include "ros/ros.h"
#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/CameraInfo.h"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

sensor_msgs::CameraInfo camera_info;

ros::Publisher debug_pub;

void imageCallback(const sensor_msgs::CompressedImageConstPtr& msg)
{
    cv::Mat image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (image.empty())
    {
        ROS_ERROR("Failed to decode image");
        return;
    }   
    sensor_msgs::ImagePtr out = cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();

    debug_pub.publish(*out);

}

void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
{
  camera_info = *msg;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "camera_node");
  ros::NodeHandle n;

  debug_pub = n.advertise<sensor_msgs::Image>("/ft/debug/image", 1);

  ros::Subscriber camera_sub = n.subscribe("/hk_camera/image_raw/compressed", 1, imageCallback);

  ros::Subscriber camera_sub2 = n.subscribe("/hk_camera/camera_info", 1, cameraInfoCallback);

  ros::spin();

  return 0;
}