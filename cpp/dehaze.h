#ifndef DEHAZE_H
#define DEHAZE_H

#include <opencv2/opencv.hpp>

cv::Mat dehaze(cv::Mat img,  size_t patchSize = 15 );

cv::Mat dehaze( const char * filename, size_t patchSize = 15 );

void debugImageInfo( IplImage * image );

// 得到图片的 Dark Channel .
cv::Mat darkChannelOfImage( cv::Mat image, size_t patchSize );
cv::Rect hazestRectOfImage( cv::Mat img, float ratio );
cv::Vec3b airlightOfImage( cv::Mat originalImage, cv::Mat darkChannel, float ratio );
cv::Mat transimissionOfImage(cv::Mat img, cv::Vec3b color, float w = 0.95);

cv::Mat recoverImage( cv::Mat img, cv::Vec3b airlight, cv::Mat t, float t0 );
#endif
