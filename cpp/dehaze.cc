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
using std::max;
using cv::Rect;
using cv::rectangle;
using cv::Scalar;
using cv::Range;
using cv::sortIdx;

uchar min( Vec3b vec ) {
    return min(vec[0], min(vec[1], vec[2]));
}

uchar max( Vec3b vec ) {
    return max(vec[0], max(vec[1], vec[2]));
}

Mat dehaze( const char * filename, size_t patchSize ){
    cout << "Load image from: " << filename << endl;
    Mat img = imread(filename);
    if( img.data != NULL ){
        cout << "Loaded image successed." << endl;
        return dehaze(img, patchSize);
    } else {
        cout << "Load image error, check whether image file exists or not." << endl;
    }
    return Mat();
}

void debugImageInfo( Mat img ){
    Size size = img.size();
    cout << "Image has size :( " << size.width << ", " << size.height << " )." << endl;
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
    Vec3b airlight = airlightOfImage(img, darkChannel, 0.001);
    Mat t = transimissionOfImage(img, airlight);
    Mat dehazedImg = recoverImage( img, airlight, t, 0.1);
    return dehazedImg;
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

Rect hazestRectOfImage( Mat img, float ratio ) {
    Size imgSize = img.size();
    int nPoints = imgSize.width * imgSize.height;
    int nSelectedPoints = nPoints * ratio;
    Mat newimg = img.reshape( 0, 1 ); 
    Mat sortedIndice;
    sortIdx(newimg, sortedIndice, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
    Mat hazestPoints = Mat(sortedIndice, Range::all(), Range(0, nSelectedPoints));
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
    return Rect(minCol, minRow, maxCol - minCol,  maxRow - minRow);
}

Vec3b brightestPixel( Mat image, Rect rect ) {
    Mat_<Vec3b> img = image;
    uchar maxBrightness = 0;
    Size imgSize = image.size();
    int maxRow, maxCol;
    for( int row = 0; row < imgSize.height; ++row ) {
        for( int col = 0; col < imgSize.width; ++col ) {
            if( maxBrightness < max(img(row, col)) ){
                maxBrightness = max(img(row, col));
                maxRow = row; maxCol = col;
            }
        }
    }
    return img(maxRow, maxCol);
}

Vec3b airlightOfImage( Mat originalImage, Mat darkChannel, float ratio ) {
    Rect rect = hazestRectOfImage( darkChannel, ratio );
    Vec3b airlight = brightestPixel(originalImage, rect);
    cout << "Airlight:" << airlight << endl;
    return airlight;
}

Mat transimissionOfImage(Mat img, Vec3b color, float w){
    Size imgSize = img.size();
    Mat_<float> transmission = Mat_<float>(imgSize.height, imgSize.width);
    Mat_<Vec3b> img_ = img;
    float r,g,b, minChn;
    for( int row = 0; row < imgSize.height; ++row ){
        for( int col = 0; col < imgSize.width; ++col ){
            b = img_(row, col)[0] / float(color[0]);
            g = img_(row, col)[1] / float(color[1]);
            r = img_(row, col)[2] / float(color[2]);
            minChn = min(b, min(g, r));
            transmission(row, col) = 1 - minChn * w;
        }
    }
    return transmission;
}

Mat recoverImage( Mat img, Vec3b airlight, Mat t, float t0 ){
    Size imgSize = img.size();
    Mat_<Vec3b> img_ = img;
    Mat_<float> t_ = t;
    Mat_<Vec3b> dehazedImg = Mat::zeros( imgSize, CV_8UC3 );
    for( int row = 0; row < imgSize.height; ++row ){
        for( int col = 0; col < imgSize.width; ++col) {
            dehazedImg(row, col) = airlight - ( airlight - img_(row, col)  ) / max(t_(row, col), t0);
        }
    }
    return dehazedImg;
}
