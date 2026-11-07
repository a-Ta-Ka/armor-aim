/*
#include "ros/ros.h"
#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/CameraInfo.h"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

cv::Mat K, D;
int img_width = 0;
int img_height = 0;

ros::Publisher debug_pub;

void imageCallback(const sensor_msgs::CompressedImageConstPtr& msg)
{
  cv::Mat image=cv::imdecode(cv::Mat(msg->data),cv::IMREAD_COLOR);

  if(image.empty())
  {
    ROS_ERROR("Failed to decode image");
    return;
  }

  sensor_msgs::ImagePtr out = cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();

  debug_pub.publish(*out);
}

void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
{
  if(msg->K[0] == 0.0)
  {
    ROS_WARN("camera info K is empty,skip");
    return;
  }

  K = cv::Mat(3, 3, CV_64F, (void*)msg->K.data()).clone();
  D.release();
  if (!msg->D.empty())
  {
    D = cv::Mat(msg->D.size(), 1, CV_64F, (void *)msg->D.data()).clone();
  }

  img_width = msg->width;
  img_height = msg->height;

  ROS_INFO("camera info %dx%d, model=%s, D size=%zu", img_width, img_height, msg->distortion_model.c_str(), msg->D.size());
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "camera_node");
  ros::NodeHandle n;

  std::string debug_topic;
  std::string image_topic;
  std::string camera_info_topic;
  int queue_size;

  n.param<std::string>("debug_topic",debug_topic, "/ft/debug/image"); 
  n.param<std::string>("image_topic",image_topic, "/hk_camera/image_raw/compressed");
  n.param<std::string>("camera_info_topic",camera_info_topic, "/hk_camera/camera_info");
  n.param<int>("queue_size",queue_size, 1);

  debug_pub = n.advertise<sensor_msgs::Image>(debug_topic, queue_size);

  ros::Subscriber camera_sub = n.subscribe(image_topic, queue_size, imageCallback);

  ros::Subscriber camera_sub2 = n.subscribe(camera_info_topic, queue_size, cameraInfoCallback);

  ros::spin();

  return 0;
}
*/


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
  // 零拷贝：直接共享 msg 里的像素内存，不复制
  cv_bridge::CvImageConstPtr cv_ptr;

  cv_ptr = cv_bridge::toCvShare(msg, "bgr8");

  // cv_ptr->image 就是 cv::Mat，和 msg 共享同一块内存
  const cv::Mat& image = cv_ptr->image;

  // 用原 header（stamp + frame_id）
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
  // 注意：话题名不带 /compressed，image_transport 自动找合适的版本
  image_transport::Subscriber sub =
      it.subscribe(image_topic, queue_size, imageCallback,hints);

  ros::Subscriber sub_info =
      n.subscribe(camera_info_topic, queue_size, cameraInfoCallback);

  ros::spin();
  return 0;
}