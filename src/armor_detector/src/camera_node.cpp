#include "ros/ros.h"
#include "sensor_msgs/CameraInfo.h"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/transport_hints.h>

cv::Mat K, D;
int img_width = 0;
int img_height = 0;


image_transport::Publisher debug_pub;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImageConstPtr cv_ptr;

  cv_ptr = cv_bridge::toCvShare(msg, "bgr8");

  const cv::Mat& image = cv_ptr->image;

  sensor_msgs::ImagePtr out = cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();
  debug_pub.publish(out);
}

void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
{
  if (msg->K[0] == 0.0)
  {
    ROS_WARN("K是空的");
    return;
  }

  K = cv::Mat(3, 3, CV_64F, (void*)msg->K.data()).clone();

  D.release();
  if (!msg->D.empty())
  {
    D = cv::Mat(msg->D.size(), 1, CV_64F, (void*)msg->D.data()).clone();
  }

  img_width = msg->width;
  img_height = msg->height;

  ROS_INFO("像素是%dx%d, 畸变模型=%s",
           img_width, img_height, msg->distortion_model.c_str());
}

int main(int argc, char **argv)
{
  setlocale(LC_ALL, ""); 

  ros::init(argc, argv, "camera_node");
  ros::NodeHandle n;

  image_transport::ImageTransport it(n);

  std::string image_topic;
  std::string camera_info_topic;
  std::string debug_topic;
  int queue_size;

  n.param<std::string>("image_topic", image_topic, "/hk_camera/image_raw");
  n.param<std::string>("camera_info_topic", camera_info_topic, "/hk_camera/camera_info");
  n.param<std::string>("debug_topic", debug_topic, "/ft/debug/image");
  n.param<int>("queue_size", queue_size, 1);

  debug_pub = it.advertise(debug_topic, queue_size);

  image_transport::TransportHints hints("compressed", ros::TransportHints(), n);

  image_transport::Subscriber sub =
      it.subscribe(image_topic, queue_size, imageCallback,hints);

  ros::Subscriber sub_info =
      n.subscribe(camera_info_topic, queue_size, cameraInfoCallback);

  ros::spin();
  return 0;
}