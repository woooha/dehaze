#ifndef DEHAZE_H
#define DEHAZE_H

#include <opencv2/opencv.hpp>

cv::Mat dehaze(cv::Mat img,  size_t patchSize = 15 );

cv::Mat dehaze( const char * filename, size_t patchSize = 15 );

void debugImageInfo( IplImage * image );

// 得到图片的 Dark Channel .
cv::Mat darkChannelOfImage( cv::Mat image, size_t patchSize );
cv::Rect hazestRectOfImage( cv::Mat img, float ratio );
#endif
