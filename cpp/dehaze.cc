#include "dehaze.h"
#include <iostream>
#include <cmath>
#include <algorithm>

using cv::Mat;
using cv::Mat_;
using cv::imread;
using cv::Size;
using cv::Vec3b;
using cv::Vec;
using std::cout;
using std::endl;
using std::min;

uchar min( Vec3b vec ) {
    return min(vec[0], min(vec[1], vec[2]));
}

Mat dehaze( const char * filename, size_t patchSize ){
    std::cout << "Load image from: " << filename << std::endl;
    Mat img = imread(filename);
    if( img.data != NULL ){
        std::cout << "Loaded image successed." << std::endl;
        return dehaze(img, patchSize);
    } else {
        std::cout << "Load image error, check whether image file exists or not." << std::endl;
    }
    return Mat();
}

void debugImageInfo( Mat img ){
    Size size = img.size();
    std::cout << "Image has size :( " << size.width << ", " << size.height << " )." << std::endl;
}

uchar darkValue( Mat &img, int row, int col, size_t patchSize ) {
    uchar minValue = 255;
    int patchRowBegin = row - patchSize / 2, patchRowEnd = patchRowBegin + patchSize;
    int patchColBegin = col - patchSize / 2, patchColEnd = patchColBegin + patchSize;
    patchRowBegin = patchRowBegin >= 0 ? patchRowBegin : 0;
    patchColBegin = patchColBegin >= 0 ? patchColBegin : 0;
    patchRowEnd = patchRowEnd < img.rows ? patchRowEnd : img.rows;
    patchColEnd = patchColEnd < img.cols ? patchColEnd : img.cols;
    uchar * rowPtr;
    for( int nRow = patchRowBegin; nRow < patchRowEnd; ++nRow ) {
        rowPtr = img.ptr<uchar>(nRow);
        for( int nCol = patchColBegin; nCol < patchColEnd; ++nCol) {
            if( rowPtr[nCol] < minValue ){
                minValue = rowPtr[nCol];
            }
        }
    }
    return minValue;
}

Mat dehaze( Mat img, size_t patchSize ) {
    debugImageInfo( img );
    // 得到图片的 Dark Channel.
    Mat darkChannel = darkChannelOfImage( img, patchSize );
    cv::Rect rect = hazestRectOfImage( darkChannel, 0.001 );
    cv::rectangle(darkChannel, rect, cv::Scalar(255) );
    return darkChannel;
}

Mat darkChannelOfImage( Mat image, size_t patchSize ) {
    Size imgSize = image.size();
    //生成一个灰度图, nChannels == 1
    Mat darkChannelMat = Mat::zeros( imgSize, CV_8UC1 );
    Mat minChannelMat = Mat::zeros( imgSize, CV_8UC1 );
    Mat_<Vec3b> img = image;

    //首先算出来这个图片最小 Channel 的颜色, min_c ( min_patch() ) == min_patch( min_c() )
    uchar * rowPtr;
    for( int row = 0; row < imgSize.height; ++row ) {
        rowPtr = minChannelMat.ptr<uchar>(row);
        for( int col = 0; col < imgSize.width; ++col ) {
            rowPtr[col] = min(img(row, col)); 
        }
    }

    //针对图片的每一点，取这一点的 Dark Value.
    uchar * oldRowPtr;
    for( int row = 0; row < imgSize.height; ++row ) {
        rowPtr = darkChannelMat.ptr<uchar>(row);
        oldRowPtr = minChannelMat.ptr<uchar>(row);
        for( int col = 0; col < imgSize.width; ++col ) {
            rowPtr[col] = darkValue(minChannelMat, row, col, patchSize);
        }
    }
    return darkChannelMat;
}

cv::Rect hazestRectOfImage( Mat img, float ratio ) {
    Size imgSize = img.size();
    int nPoints = imgSize.width * imgSize.height;
    int nSelectedPoints = nPoints * ratio;
    Mat newimg = img.reshape( 0, 1 ); 
    Mat sortedIndice;
    cv::sortIdx(newimg, sortedIndice, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
    Mat hazestPoints = Mat(sortedIndice, cv::Range::all(), cv::Range(0, nSelectedPoints));
    cout << hazestPoints << endl;
    int minRow = imgSize.height, minCol = imgSize.width, maxRow = 0, maxCol = 0;
    int * p = hazestPoints.ptr<int>(0);
    for( int i = 0; i < hazestPoints.cols; ++i ){
        int pos = p[i];
        int row = pos / imgSize.width;
        int col = pos % imgSize.width;

        if( row > maxRow ){
            maxRow = row;
        } else if( row < minRow ) {
            minRow = row;
        }

        if( col > maxCol ){
            maxCol = col;
        } else if( col < minCol ){
            minCol = col;
        }
    }
    cout << "The hazest region is:" << endl;
    cout << "( " << minRow << ", " << minCol << " ), " << "( " << maxRow << ", " << maxCol << " )" <<endl;
    return cv::Rect(minCol, minRow, maxCol - minCol,  maxRow - minRow);
}

void brightestPixel( Mat img, cv::Rect rect ) {

}

void airlightOfImage( Mat darkChannel, float ratio ) {
    cv::Rect rect = hazestRectOfImage( darkChannel, ratio );
}

