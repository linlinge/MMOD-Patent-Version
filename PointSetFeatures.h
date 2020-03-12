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
        void   ComputeMahalanobisDistance(Eigen::MatrixXd v, Eigen::MatrixXd S);
        // Feature 02: Quadric Surface Fitting
        double Poly33(pcl::PointCloud<PointType>::Ptr cloud);

        // Feature 03: Thin Plane
        int    ThinPlaneCounter(pcl::PointCloud<PointType>::Ptr cloud);
        double ThinPlaneAngle(pcl::PointCloud<PointType>::Ptr cloud);
        double ThinPlaneProjector(pcl::PointCloud<PointType>::Ptr cloud);
};

// Different Features of Point Set 
class PointSetFeatures
{
    public: 
        Table rst_CentroidAndCentre_;
        Table rst_density_;
        vector<double> PLOF_;

        // Feature 01: knn Normal Angle
        int flag_knnNormalAngle_=0;
        Table rst_knnNormalAngle_;
        void knnNormalAngle(pcl::PointCloud<PointType>::Ptr cloud);

        // Feature 02: Different Kinds of Distance
        int flag_Mahalanobis_=0;
        Table rst_Mahalanobis_;
        void ApplyMahalanobis(pcl::PointCloud<PointType>::Ptr cloud);

        int flag_StandardizedEuclideanDistance_=0;
        // Table rst_StandardizedEuclideanDistance_；
        void ApplyStandardizedEuclideanDistance(pcl::PointCloud<PointType>::Ptr cloud,int K=31);

        int flag_StandardDistance_=0;
        Table rst_StandardDistance_;
        void ApplyStandardDistance(pcl::PointCloud<PointType>::Ptr cloud);

        // Feature 03: Minor Eigenvalue
        int flag_MinorEigenvalue_=0;
        Table rst_MinorEigenvalue_; 
        void ApplyMinorEigenvalue(pcl::PointCloud<PointType>::Ptr cloud, int K=35);

        int flag_EigenvalueRatio_=0;
        Table rst_EigenvalueRatio_;
        void ApplyEigenvalueRatio(pcl::PointCloud<PointType>::Ptr cloud, int K=35);

        // Feature 04: Qudratic Surface Fitting
        int flag_QuadricSurfaceFitting_=0;
        Table rst_QuadricSurfaceFitting_;
        void ApplyQuadricSurfaceFitting(pcl::PointCloud<PointType>::Ptr cloud);

        // Feature 05: Thin Plane
        int flag_ThinPlaneCounter_=0;
        Table rst_ThinPlaneCounter_;
        void ApplyThinPlaneCounter(pcl::PointCloud<PointType>::Ptr cloud,int K=25);

        int flag_ThinPlaneAngle_=0;
        Table rst_ThinPlaneAngle_;
        void ApplyThinPlaneAngle(pcl::PointCloud<PointType>::Ptr cloud,int K=25);

        int flag_ThinPlaneProjection_=0;
        Table rst_ThinPlaneProjection_; 
        void ApplyThinPlaneProjection(pcl::PointCloud<PointType>::Ptr cloud,int K=25);


        // Feature 06: Loop
        int flag_Loop_=0;
        Table rst_Loop_;
        void ApplyLoop(pcl::PointCloud<PointType>::Ptr cloud,int K=31);

        // Feature 06: Centroid and Centre
        int flag_CentroidAndCentre_=0;
        void ApplyCentroidAndCentre(pcl::PointCloud<PointType>::Ptr cloud);

        // Feature 07: Density
        int flag_density_=0;
        void ApplyDensity(pcl::PointCloud<PointType>::Ptr cloud,double lamda=5);

        // Write Result to csv
        void Write(string path,pcl::PointCloud<PointType>::Ptr cloud);

        // Feature 08: 
        int flag_slope_=0;
        int flag_gap_=0;
        int flag_pulse_=0;
        Table rst_slope_,rst_gap_max_,rst_gap_var_,rst_pulse_;
        void ApplykNN(pcl::PointCloud<PointType>::Ptr cloud, int K=80,string mode="gap");
};