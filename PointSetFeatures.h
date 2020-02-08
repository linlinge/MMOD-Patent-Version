#pragma once
#include "PCLExtend.h"
#include <Eigen/Dense>
#include "Table.h"
#include "Color.h"
#include "Statistics.h"
#include <numeric>

// Normal Angle
class Feature
{
    public:
        // Variables
        Eigen::MatrixXd dat_;
        Eigen::MatrixXd cov_;
        Eigen::MatrixXd mean_;
        Eigen::MatrixXd Mdist_;
        Eigen::MatrixXd P_;

        // One Point
        double GetNormalAngle(pcl::PointCloud<PointType>::Ptr cloud);        
        void ComputeMahalanobisDistance(Eigen::MatrixXd v, Eigen::MatrixXd S);
        // Quadric Surface Fitting
        double Poly33(pcl::PointCloud<PointType>::Ptr cloud);
        //
        int KnnPlaneCounter(pcl::PointCloud<PointType>::Ptr cloud);
        double KnnPlaneProjector(pcl::PointCloud<PointType>::Ptr cloud);
};

// Different Features of Point Set 
class PointSetFeatures
{
    public:
        Table rst_KnnPlane_cnt_;
        Table rst_MinorEigenvalue_;
        Table rst_KnnPlaneProjection_;
        Table rst_CentroidAndCentre_;
        Table rst_StandardizedEuclideanDistance_;
        Table rst_density_;
        Table rst_;
        pcl::PointCloud<PointType>::Ptr cloud_;
        pcl::PointCloud<PointType>::Ptr cloud_trusted_;
        pcl::PointCloud<PointType>::Ptr cloud_untrusted_;
        vector<double> PLOF_;

        // Construction
        PointSetFeatures(pcl::PointCloud<PointType>::Ptr cloud);

        // Feature 01: knn Normal Angle
        int flag_knnNormaAngle_=0;
        void knnNormaAngle(pcl::PointCloud<PointType>::Ptr cloud);

        // Feature 02: Different Kinds of Distance
        int flag_Mahalanobis_=0;
        void ApplyMahalanobis(pcl::PointCloud<PointType>::Ptr cloud);
        int flag_StandardizedEuclideanDistance_=0;
        void ApplyStandardizedEuclideanDistance(pcl::PointCloud<PointType>::Ptr cloud,int K=31);
        int flag_StandardDistance_=0;
        void ApplyStandardDistance(pcl::PointCloud<PointType>::Ptr cloud);

        // Feature 03: Minor Eigenvalue
        int flag_MinorEigenvalue_=0;
        void ApplyMinorEigenvalue(pcl::PointCloud<PointType>::Ptr cloud, int K=35);
        int flag_EigenvalueRatio_=0;
        void ApplyEigenvalueRatio(pcl::PointCloud<PointType>::Ptr cloud, int K=35);

        // Feature 04: Qudratic Surface Fitting
        int flag_QuadricSurfaceFitting_=0;
        void ApplyQuadricSurfaceFitting(pcl::PointCloud<PointType>::Ptr cloud);

        // Feature 05: Knn Plane
        int flag_KnnPlane_=0;
        void ApplyKnnPlane(pcl::PointCloud<PointType>::Ptr cloud,int K=25);
        int flag_KnnPlaneProjection_=0;
        void ApplyKnnPlaneProjection(pcl::PointCloud<PointType>::Ptr cloud,int K=25);
        // 
        void ApplyPLOF(int K=31);
        void ApplynPLOF();

        // Feature 06: Centroid and Centre
        int flag_CentroidAndCentre_=0;
        void ApplyCentroidAndCentre(pcl::PointCloud<PointType>::Ptr cloud);

        // Feature 07: Density
        int flag_density_=0;
        void ApplyDensity(pcl::PointCloud<PointType>::Ptr cloud,double lamda=5);

        // Write Result to csv
        void Write(string path,pcl::PointCloud<PointType>::Ptr cloud);
};