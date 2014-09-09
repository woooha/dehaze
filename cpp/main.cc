#include <iostream>
#include <string>
#include <google/gflags.h>
#include "dehaze.h"

DEFINE_string(image, "", "Image path to processed.");
DEFINE_string(output, "", "Image to output.");
DEFINE_int32(patch_size, 15, "Patch size to compute dark channel.");

int main( int argc, char ** argv ){
    std::string usage("This program will remove the haze in the image. For example:\n");
    usage += std::string(argv[0]) + "--image ./image.jpg";
    ::google::SetUsageMessage(usage);
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    cv::Mat dehazedImage = dehaze( FLAGS_image.c_str(), FLAGS_patch_size); 
    cv::imshow("darkchannel", dehazedImage);
    cv::waitKey();
    //cv::imwrite(FLAGS_output.c_str(), dehazedImage);
    return 0;
}
