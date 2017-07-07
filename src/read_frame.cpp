// OpenCV
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d/features2d.hpp>

// ros
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Twist.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Int16.h>

/**image transport used to receive rgbd camera image*/
image_transport::Subscriber rgb_image_sub;
image_transport::Subscriber depth_image_sub;
cv::Mat g_rgb_image;
cv::Mat g_depth_image;

/*rgb and depth image callback*/
void rgbImageCallback(const sensor_msgs::Image::ConstPtr msg);
void depthImageCallback(const sensor_msgs::Image::ConstPtr msg);

int main(int argc, char **argv)
{
  ros::init(argc, argv, "read_frame");
  ros::NodeHandle node;
  ros::Rate loop_rate(50);

  /*subscribe to rgb and depth image of rgbd camera*/
  image_transport::ImageTransport image_transport(node);
  rgb_image_sub= image_transport.subscribe("/camera/rgb/image_raw", 1,
                                           rgbImageCallback);
  depth_image_sub= image_transport.subscribe("/camera/depth/image", 1,
                                             depthImageCallback);

  /*create windows to show rgb and depth image*/
  cv::namedWindow("rgb_image", 1);
  cv::namedWindow("depth_image", 1);
  /*begin main loop, read image*/
  ROS_INFO_STREAM("begin to read image");
  while(ros::ok())
  {
    ros::spinOnce();
    if(!g_rgb_image.empty())
    {
      cv::imshow("rgb_image", g_rgb_image);
    }
    if(!g_depth_image.empty())
    {
      cv::imshow("depth_image", g_depth_image);
    }
    cv::waitKey(1);
  }
  return 0;
}

/*subscribe to rgb image*/
void rgbImageCallback(const sensor_msgs::Image::ConstPtr msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr=
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch(cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  g_rgb_image= cv_ptr->image;
}

/*subscribe to depth image*/
void depthImageCallback(const sensor_msgs::Image::ConstPtr msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr= cv_bridge::toCvCopy(
        msg, sensor_msgs::image_encodings::TYPE_32FC1);
  }
  catch(cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  g_depth_image= cv_ptr->image;
}