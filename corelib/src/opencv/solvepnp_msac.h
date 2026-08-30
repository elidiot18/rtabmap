/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

// This file is a modified fork of OpenCV's solvepnp.cpp / ptsetreg.cpp, adapted to
// implement MSac (M-estimator SAmple Consensus) scoring instead of RANSAC, optionally
// weighted by a per-correspondence 3D covariance. The original OpenCV license above
// applies to the derived work.

#ifndef RTABMAP_CORELIB_SRC_OPENCV_SOLVEPNP_MSAC_H_
#define RTABMAP_CORELIB_SRC_OPENCV_SOLVEPNP_MSAC_H_

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#if CV_MAJOR_VERSION >= 3
#include <opencv2/calib3d/calib3d_c.h>
#endif

#include "solvepnp.h"
#include "rtabmap/core/Transform.h"

using namespace cv;

namespace cv_custom {


std::vector<float> computeMahalanobisReprojErrors(
   const std::vector<cv::Point3f>& opoints,
   const std::vector<cv::Point2f>& ipoints,
   const cv::Mat & cameraMatrix,
   const cv::Mat & distCoeffs,
   const cv::Mat & rvec,
   const cv::Mat & tvec,
   const std::vector<cv::Matx33f> & cov3D,
   float pixelVariance,
   float squaredErrorThreshold,
   std::vector<int> & inliers);

bool solvePnPMsac(cv::InputArray objectPoints, cv::InputArray imagePoints,
                  cv::InputArray cameraMatrix, cv::InputArray distCoeffs,
                  const std::vector<cv::Matx33f>& covariances3A,
                  cv::OutputArray rvec, cv::OutputArray tvec,
                  bool useExtrinsicGuess = false, int iterationsCount = 100,
                  float chi2Threshold = 2.44765f, float confidence = 0.99, 
                  float pixelVariance = 1.0f,
                  cv::OutputArray inliers = cv::noArray(), int flags = cv::SOLVEPNP_ITERATIVE);

void solvePnPMsacRefineLM(
    cv::Mat & rvec,
    cv::Mat & tvec,
    const std::vector<cv::Point3f> & opoints_inliers,
    const std::vector<cv::Point2f> & ipoints_inliers,
    const std::vector<cv::Matx33f> & cov_inliers,
    float pixelVariance,
    const cv::Mat & cameraMatrix,
    const cv::Mat & distCoeffs,
    int maxIterations = 20);

cv::Ptr<cv3::PointSetRegistrator> createMSACPointSetRegistrator(
    const cv::Ptr<cv3::PointSetRegistrator::Callback>& cb,
    int modelPoints, double threshold, double confidence,
    int maxIters);

} // namespace cv_custom

#endif /* RTABMAP_CORELIB_SRC_OPENCV_SOLVEPNP_MSAC_H_ */