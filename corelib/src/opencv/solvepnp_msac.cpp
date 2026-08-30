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

#include <iostream>
#include <limits>
#include "solvepnp_msac.h"

#include "rtabmap/utilite/UMath.h"
#include "rtabmap/utilite/ULogger.h"

using namespace cv;

namespace cv_custom
{

const float CHI2_95_UNSQUARED = 2.44765f;

cv::Matx22d computeWhiteningMatrix(
    const cv::Point3f& pt3d, const cv::Matx33d& R33, const cv::Matx31d& tvec31, 
    double fx, double fy, const cv::Matx33f& cov3D, float pixelVariance) 
{
    Matx31d XA(pt3d.x, pt3d.y, pt3d.z);
    Matx31d XB = R33 * XA + tvec31;
    double Z = std::max(XB(2), 1e-5); 

    Matx23d J_pi(fx/Z, 0.0, -fx*XB(0)/(Z*Z), 0.0, fy/Z, -fy*XB(1)/(Z*Z));
    Matx23d J_XA = J_pi * R33;
    Matx22d cov2D = J_XA * Matx33d(cov3D) * J_XA.t();
    cov2D(0,0) += pixelVariance; cov2D(1,1) += pixelVariance;
    double det = std::max(cov2D(0,0)*cov2D(1,1) - cov2D(0,1)*cov2D(1,0), 1e-12);
        
    return Matx22d(cov2D(1,1)/det, -cov2D(0,1)/det, -cov2D(1,0)/det, cov2D(0,0)/det);
}

class PnPMsacCallback : public cv3::PointSetRegistrator::Callback
{

public:

    PnPMsacCallback(Mat _cameraMatrix=Mat(3,3,CV_64F), Mat _distCoeffs=Mat(4,1,CV_64F), 
            const std::vector<Matx33f>& _cov3D=std::vector<Matx33f>(), float _pixelVariance=1.0f, int _flags=CV_ITERATIVE,
            bool _useExtrinsicGuess=false, Mat _rvec=Mat(), Mat _tvec=Mat() )
        : cameraMatrix(_cameraMatrix), distCoeffs(_distCoeffs), cov3D(_cov3D),
          pixelVariance(_pixelVariance), flags(_flags), useExtrinsicGuess(_useExtrinsicGuess),
          rvec(_rvec), tvec(_tvec) {}

    /* Pre: True */
    /* Post: compute _model with given points an return number of found models */
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const override
    {
        Mat opoints = _m1.getMat(), ipoints = _m2.getMat();

        bool correspondence = solvePnP( _m1, _m2, cameraMatrix, distCoeffs,
                                            rvec, tvec, useExtrinsicGuess, flags );

        Mat _local_model;
        hconcat(rvec, tvec, _local_model);
        _local_model.copyTo(_model);

        return correspondence;
    }

    /* Pre: True */
    /* Post: fill _err with projection errors */
    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const override
    {

        Mat opoints = _m1.getMat(), ipoints = _m2.getMat(), model = _model.getMat();

        int i, count = opoints.checkVector(3);
        Mat _rvec = model.col(0);
        Mat _tvec = model.col(1);


        Mat projpoints(count, 2, CV_32FC1);
        projectPoints(opoints, _rvec, _tvec, cameraMatrix, distCoeffs, projpoints);

        Mat R;
        Rodrigues(_rvec, R);
        Matx33d R33; R.convertTo(R33, CV_64F);
        Matx31d tvec31(_tvec.at<double>(0), _tvec.at<double>(1), _tvec.at<double>(2));

        double fx = cameraMatrix.at<double>(0,0);
        double fy = cameraMatrix.at<double>(1,1);

        const Point3f* pt3d = opoints.ptr<Point3f>();
        const Point2f* pt2d = ipoints.ptr<Point2f>();
        const Point2f* pproj = projpoints.ptr<Point2f>();

        _err.create(count, 1, CV_32FC1);
        float* err = _err.getMat().ptr<float>();

        for ( i = 0; i < count; ++i)
        {
            Matx22d cov2D_inv;
            if(!cov3D.empty())
            {
                cov2D_inv = computeWhiteningMatrix(pt3d[i], R33, tvec31, fx, fy, cov3D[i], pixelVariance);
            }
            else
            {
                cov2D_inv = (pixelVariance > 0.0f) ? Matx22d(1.0/pixelVariance, 0.0, 0.0, 1.0/pixelVariance) : Matx22d(1.0, 0.0, 0.0, 1.0);
            }

            double ex = pt2d[i].x - pproj[i].x;
            double ey = pt2d[i].y - pproj[i].y;

            // Cheirality violation: without this the point would score as the best inlier.
            const cv::Matx31d XBchk = R33 * cv::Matx31d(pt3d[i].x, pt3d[i].y, pt3d[i].z) + tvec31;
            if(XBchk(2) <= 1e-3)
            {
                err[i] = std::numeric_limits<float>::max();
                continue;
            }

            err[i] = static_cast<float>(ex * ex * cov2D_inv(0,0) + 2.0 * ex * ey * cov2D_inv(0,1) + ey * ey * cov2D_inv(1,1));
        }

    }


    Mat cameraMatrix;
    Mat distCoeffs;
    std::vector<Matx33f> cov3D;
    float pixelVariance;
    int flags;
    bool useExtrinsicGuess;
    Mat rvec;
    Mat tvec;
};

bool solvePnPMsac(InputArray _opoints, InputArray _ipoints,
                  InputArray _cameraMatrix, InputArray _distCoeffs,
                  const std::vector<cv::Matx33f>& covariances3A,
                  OutputArray _rvec, OutputArray _tvec, bool useExtrinsicGuess,
                  int iterationsCount, float chi2Threshold, float confidence, 
                  float pixelVariance, OutputArray _inliers, 
                  int flags)
{

    Mat opoints0 = _opoints.getMat(), ipoints0 = _ipoints.getMat();
    Mat opoints, ipoints;
    if( opoints0.depth() == CV_64F || !opoints0.isContinuous() )
        opoints0.convertTo(opoints, CV_32F);
    else
        opoints = opoints0;
    if( ipoints0.depth() == CV_64F || !ipoints0.isContinuous() )
        ipoints0.convertTo(ipoints, CV_32F);
    else
        ipoints = ipoints0;

    int npoints = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
    CV_Assert( npoints >= 0 && npoints == std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)) );

    CV_Assert(opoints.isContinuous());
    CV_Assert(opoints.depth() == CV_32F || opoints.depth() == CV_64F);
    CV_Assert((opoints.rows == 1 && opoints.channels() == 3) || opoints.cols*opoints.channels() == 3);
    CV_Assert(ipoints.isContinuous());
    CV_Assert(ipoints.depth() == CV_32F || ipoints.depth() == CV_64F);
    CV_Assert((ipoints.rows == 1 && ipoints.channels() == 2) || ipoints.cols*ipoints.channels() == 2);

    Mat rvec = useExtrinsicGuess ? _rvec.getMat() : Mat(3, 1, CV_64FC1);
    Mat tvec = useExtrinsicGuess ? _tvec.getMat() : Mat(3, 1, CV_64FC1);
    Mat cameraMatrix = _cameraMatrix.getMat(), distCoeffs = _distCoeffs.getMat();

    int model_points = 6;
    int ransac_kernel_method = CV_EPNP;

    if( npoints == 4 )
    {
        model_points = 4;
        ransac_kernel_method = CV_P3P;
    }

    Ptr<cv3::PointSetRegistrator::Callback> cb; // pointer to callback
    // Cloned: cv::Mat assignment shares the buffer, and runKernel() would overwrite the guess.
    cb = Ptr<PnPMsacCallback>(new PnPMsacCallback( cameraMatrix, distCoeffs, covariances3A, pixelVariance, ransac_kernel_method, useExtrinsicGuess, rvec.clone(), tvec.clone()));

    double param1 =  chi2Threshold;                   // chi2 threshold
    double param2 = confidence;                       // confidence
    int param3 = iterationsCount;                     // number maximum iterations

    Mat _local_model(3, 2, CV_64FC1);
    Mat _mask_local_inliers(1, opoints.rows, CV_8UC1);

    // call MSAC
    int result = createMSACPointSetRegistrator(cb, model_points,
        param1, param2, param3)->run(opoints, ipoints, _local_model, _mask_local_inliers);

    if( result > 0 )
    {
        std::vector<Point3d> opoints_inliers;
        std::vector<Point2d> ipoints_inliers;
        opoints.convertTo(opoints_inliers, CV_64F);
        ipoints.convertTo(ipoints_inliers, CV_64F);

        const uchar* mask = _mask_local_inliers.ptr<uchar>();
        int npoints1 = cv3::compressElems(&opoints_inliers[0], mask, 1, npoints);
        cv3::compressElems(&ipoints_inliers[0], mask, 1, npoints);

        opoints_inliers.resize(npoints1);
        ipoints_inliers.resize(npoints1);

        Mat rvec_refined = _local_model.col(0).clone();
        Mat tvec_refined = _local_model.col(1).clone();
        result = solvePnP(opoints_inliers, ipoints_inliers, cameraMatrix,
                          distCoeffs, rvec_refined, tvec_refined, true, flags == cv::SOLVEPNP_P3P ? cv::SOLVEPNP_EPNP : flags) ? 1 : -1;
        if( result > 0 )
        {
            rvec_refined.copyTo(rvec);
            tvec_refined.copyTo(tvec);
        }
    }

    if( result <= 0 || _local_model.rows <= 0)
    {
    	rvec.copyTo(_rvec);    // output rotation vector
        tvec.copyTo(_tvec);    // output translation vector

        if( _inliers.needed() )
            _inliers.release();

        return false;
    }
    else
    {
        // Output the refined pose, not _local_model (only the best minimal-sample hypothesis).
        rvec.copyTo(_rvec);    // output rotation vector
        tvec.copyTo(_tvec);    // output translation vector
    }

    if(_inliers.needed())
    {
        Mat _local_inliers;
        for (int i = 0; i < npoints; ++i)
        {
            if((int)_mask_local_inliers.at<uchar>(i) != 0) // inliers mask
                _local_inliers.push_back(i);    // output inliers vector
        }
        _local_inliers.copyTo(_inliers);
    }
    return true;
}


class MSACPointSetRegistrator : public cv3::PointSetRegistrator
{
public:
    MSACPointSetRegistrator(const Ptr<cv3::PointSetRegistrator::Callback>& _cb=Ptr<cv3::PointSetRegistrator::Callback>(),
                              int _modelPoints=0, double _threshold=0, double _confidence=0.99, int _maxIters=1000)
    : cb(_cb), modelPoints(_modelPoints), threshold(_threshold), confidence(_confidence), maxIters(_maxIters)
    {
        checkPartialSubsets = false;
    }

    int evaluateModel( const Mat& m1, const Mat& m2, const Mat& model, Mat& err, Mat& mask, double thresh, float& cost ) const
    {
        cb->computeError( m1, m2, model, err );
        mask.create(err.size(), CV_8U);

        CV_Assert( err.isContinuous() && err.type() == CV_32F && mask.isContinuous() && mask.type() == CV_8U);
        const float* errptr = err.ptr<float>();
        uchar* maskptr = mask.ptr<uchar>();
        float t = (float)(thresh*thresh);
        int i, n = (int)err.total(), nz = 0;
        cost = 0;
        for( i = 0; i < n; i++ )
        {
            float sq_err = errptr[i];
            int f = sq_err <= t;
            maskptr[i] = (uchar)f;
            nz += f;
            cost += f ? sq_err : t;
        }
        return nz;
    }

    bool getSubset( const Mat& m1, const Mat& m2,
                    Mat& ms1, Mat& ms2, RNG& rng,
                    int maxAttempts=1000 ) const
    {
        cv::AutoBuffer<int> _idx(modelPoints);
        int* idx = _idx;
        int i = 0, j, k, iters = 0;
        int esz1 = (int)m1.elemSize(), esz2 = (int)m2.elemSize();
        int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
        int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
        const int *m1ptr = m1.ptr<int>(), *m2ptr = m2.ptr<int>();

        ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
        ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

        int *ms1ptr = ms1.ptr<int>(), *ms2ptr = ms2.ptr<int>();

        CV_Assert( count >= modelPoints && count == count2 );
        CV_Assert( (esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0 );
        esz1 /= sizeof(int);
        esz2 /= sizeof(int);

        for(; iters < maxAttempts; iters++)
        {
            for( i = 0; i < modelPoints && iters < maxAttempts; )
            {
                int idx_i = 0;
                for(;;)
                {
                    idx_i = idx[i] = rng.uniform(0, count);
                    
                    for( j = 0; j < i; j++ )
                        if( idx_i == idx[j] )
                            break;
                    if( j == i )
                        break;
                }
                for( k = 0; k < esz1; k++ )
                    ms1ptr[i*esz1 + k] = m1ptr[idx_i*esz1 + k];
                for( k = 0; k < esz2; k++ )
                    ms2ptr[i*esz2 + k] = m2ptr[idx_i*esz2 + k];
                if( checkPartialSubsets && !cb->checkSubset( ms1, ms2, i+1 ))
                {
                    // we may have selected some bad points;
                    // so, let's remove some of them randomly
                    i = rng.uniform(0, i+1);
                    iters++;
                    continue;
                }
                i++;
            }
            if( !checkPartialSubsets && i == modelPoints && !cb->checkSubset(ms1, ms2, i))
                continue;
            break;
        }

        return i == modelPoints && iters < maxAttempts;
    }

    bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const override
    {
        bool result = false;
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        Mat err, mask, model, bestModel, ms1, ms2;

        int iter, niters = MAX(maxIters, 1);
        int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
        int count = m1.checkVector(d1), count2 = m2.checkVector(d2), bestInliers = 0;
        float minCost = DBL_MAX;

        RNG rng((uint64)-1);

        CV_Assert( cb );
        CV_Assert( confidence > 0 && confidence < 1 );

        CV_Assert( count >= 0 && count2 == count );
        if( count < modelPoints )
            return false;

        Mat bestMask0, bestMask;

        if( _mask.needed() )
        {
            _mask.create(count, 1, CV_8U, -1, true);
            bestMask0 = bestMask = _mask.getMat();
            CV_Assert( (bestMask.cols == 1 || bestMask.rows == 1) && (int)bestMask.total() == count );
        }
        else
        {
            bestMask.create(count, 1, CV_8U);
            bestMask0 = bestMask;
        }

        if( count == modelPoints )
        {
            if( cb->runKernel(m1, m2, bestModel) <= 0 )
                return false;
            bestModel.copyTo(_model);
            bestMask.setTo(Scalar::all(1));
            return true;
        }

        for( iter = 0; iter < niters; iter++ )
        {
            int i, currentInliers, nmodels;
            if( count > modelPoints )
            {
                bool found = getSubset( m1, m2, ms1, ms2, rng, 10000 );
                if( !found )
                {
                    if( iter == 0 )
                        return false;
                    break;
                }
            }

            nmodels = cb->runKernel( ms1, ms2, model );
            if( nmodels <= 0 )
                continue;
            CV_Assert( model.rows % nmodels == 0 );
            Size modelSize(model.cols, model.rows/nmodels);

            for( i = 0; i < nmodels; i++ )
            {
                Mat model_i = model.rowRange( i*modelSize.height, (i+1)*modelSize.height );
                float currentCost = 0;
                currentInliers = evaluateModel( m1, m2, model_i, err, mask, threshold, currentCost );

                if( currentCost < minCost && currentInliers >= modelPoints )
                {
                    std::swap(mask, bestMask);
                    model_i.copyTo(bestModel);
                    minCost = currentCost;
                    bestInliers = currentInliers;
                    niters = cv3::RANSACUpdateNumIters( confidence, (double)(count - bestInliers)/count, modelPoints, niters );
                }
            }
        }

        if( minCost < DBL_MAX )
        {
            if( bestMask.data != bestMask0.data )
            {
                if( bestMask.size() == bestMask0.size() )
                    bestMask.copyTo(bestMask0);
                else
                    transpose(bestMask, bestMask0);
            }
            bestModel.copyTo(_model);
            result = true;
        }
        else
            _model.release();

        return result;
    }

    void setCallback(const Ptr<cv3::PointSetRegistrator::Callback>& _cb) override { cb = _cb; }

    Ptr<cv3::PointSetRegistrator::Callback> cb;
    int modelPoints;
    bool checkPartialSubsets;
    double threshold;
    double confidence;
    int maxIters;
};


Ptr<cv3::PointSetRegistrator> createMSACPointSetRegistrator(const Ptr<cv3::PointSetRegistrator::Callback>& _cb,
                                                         int _modelPoints, double _threshold,
                                                         double _confidence, int _maxIters)
{
    return Ptr<cv3::PointSetRegistrator>(
        new MSACPointSetRegistrator(_cb, _modelPoints, _threshold, _confidence, _maxIters));
}

class WeightedLMSolverCallback : public cv::LMSolver::Callback
{
public:
    WeightedLMSolverCallback(InputArray _opoints, InputArray _ipoints, const std::vector<Matx33f>& _cov3D, float pixelVariance,
                         Mat _K, Mat _D)
        : opoints(_opoints.getMat()), ipoints(_ipoints.getMat()), cov3D(_cov3D), pixelVariance(pixelVariance), K(_K), D(_D) {}

    bool compute(InputArray _param, OutputArray _err, OutputArray _J) const override
    {
        UASSERT(cov3D.empty() || (size_t)opoints.checkVector(3) == cov3D.size());

        Mat param = _param.getMat();
        Mat rvec = param.rowRange(0, 3);
        Mat tvec = param.rowRange(3, 6);

        int N = opoints.checkVector(3);

        bool need_err = _err.needed();
        bool need_J = _J.needed();

        Mat proj, jac_pose;
        if (need_J) {
            projectPoints(opoints, rvec, tvec, K, D, proj, jac_pose);
            _J.create(N * 2, 6, CV_64F);
        } else {
            projectPoints(opoints, rvec, tvec, K, D, proj);
        }

        Mat err, J;
        if (need_err) {
            _err.create(N * 2, 1, CV_64F);
            err = _err.getMat();
        }
        if (need_J) {
            J = _J.getMat();
        }

        Mat R;
        Rodrigues(rvec, R);
        Matx33d R33; R.convertTo(R33, CV_64F);
        Matx31d tvec31(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        double fx = K.at<double>(0,0);
        double fy = K.at<double>(1,1);

        const Point3f* pt3d = opoints.ptr<Point3f>();
        const Point2f* pt2d = ipoints.ptr<Point2f>();
        const Point2f* pproj = proj.ptr<Point2f>();

        for(int i = 0; i < N; i++) {
            Matx22d cov2D_inv;
            if(!cov3D.empty())
            {
                cov2D_inv = computeWhiteningMatrix(pt3d[i], R33, tvec31, fx, fy, cov3D[i], pixelVariance);
            }
            else
            {
                if(pixelVariance > 0.0f)
                {
                    cov2D_inv = Matx22d(1.0/pixelVariance, 0.0, 0.0, 1.0/pixelVariance);
                }
                else
                {
                    cov2D_inv = Matx22d(1.0, 0.0, 0.0, 1.0);
                }
            }

            double l00 = std::max(1e-6, std::sqrt(std::max(0.0, cov2D_inv(0,0))));
            double l10 = cov2D_inv(0,1) / l00;
            double l11 = std::sqrt(std::max(0.0, cov2D_inv(1,1) - l10*l10));
            Matx22d Lt(l00, l10, 0.0, l11); 

            if (need_err) {
                Matx21d e(pproj[i].x - pt2d[i].x, pproj[i].y - pt2d[i].y);
                Matx21d we = Lt * e;
                err.at<double>(i*2, 0) = we(0);
                err.at<double>(i*2+1, 0) = we(1);
            }

            if (need_J) {
                Matx<double, 2, 6> J_p;
                for(int k=0; k<6; k++) {
                    J_p(0, k) = jac_pose.at<double>(i*2, k);
                    J_p(1, k) = jac_pose.at<double>(i*2+1, k);
                }
                Matx<double, 2, 6> wJ = Lt * J_p;
                for(int k=0; k<6; k++) {
                    J.at<double>(i*2, k) = wJ(0, k);
                    J.at<double>(i*2+1, k) = wJ(1, k);
                }
            }
        }
        return true;
    }
    Mat opoints, ipoints;
    std::vector<Matx33f> cov3D;
    float pixelVariance;
    Mat K, D;
};

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
   std::vector<int> & inliers)
{
    int count = (int)opoints.size();
    std::vector<cv::Point2f> projpoints;
    projectPoints(opoints, rvec, tvec, cameraMatrix, distCoeffs, projpoints);

    Mat R;
    Rodrigues(rvec, R);
    Matx33d R33; R.convertTo(R33, CV_64F);
    Matx31d tvec31(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    double fx = cameraMatrix.at<double>(0,0);
    double fy = cameraMatrix.at<double>(1,1);

    inliers.resize(count, 0);
    std::vector<float> err(count);
    int oi = 0;

    for (int i = 0; i < count; ++i)
    {
        Matx22d cov2D_inv;
        if(!cov3D.empty())
        {
            cov2D_inv = computeWhiteningMatrix(opoints[i], R33, tvec31, fx, fy, cov3D[i], pixelVariance);
        }
        else
        {
            if(pixelVariance > 0.0f)
            {
                cov2D_inv = Matx22d(1.0/pixelVariance, 0.0, 0.0, 1.0/pixelVariance);
            }
            else
            {
                cov2D_inv = Matx22d(1.0, 0.0, 0.0, 1.0);
            }
        }

        double ex = ipoints[i].x - projpoints[i].x;
        double ey = ipoints[i].y - projpoints[i].y;
        
        double sq_err = ex * ex * cov2D_inv(0,0) + 2.0 * ex * ey * cov2D_inv(0,1) + ey * ey * cov2D_inv(1,1);

        if(sq_err <= squaredErrorThreshold)
        {
            inliers[oi] = i;
            err[oi++] = static_cast<float>(sq_err);
        }
    }
    inliers.resize(oi);
    err.resize(oi);
    return err;
}

// 3x9 RPY derivatives w.r.t. the flattened rotation matrix. False at the pitch=+/-90 singularity.

void solvePnPMsacRefineLM(
    cv::Mat & rvec,
    cv::Mat & tvec,
    const std::vector<cv::Point3f> & opoints_inliers,
    const std::vector<cv::Point2f> & ipoints_inliers,
    const std::vector<cv::Matx33f> & cov_inliers,
    float pixelVariance,
    const cv::Mat & cameraMatrix,
    const cv::Mat & distCoeffs,
    int maxIterations)
{
    cv::Mat param(6, 1, CV_64F);
    rvec.copyTo(param.rowRange(0, 3));
    tvec.copyTo(param.rowRange(3, 6));

    cv::Ptr<cv::LMSolver::Callback> lm_cb = cv::makePtr<cv_custom::WeightedLMSolverCallback>(
        opoints_inliers, ipoints_inliers, cov_inliers, pixelVariance, cameraMatrix, distCoeffs
    );

    cv::Ptr<cv::LMSolver> solver = cv::LMSolver::create(lm_cb, maxIterations); 
    solver->run(param);

    param.rowRange(0, 3).copyTo(rvec);
    param.rowRange(3, 6).copyTo(tvec);
}

} // namespace cv_custom