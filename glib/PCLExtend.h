#pragma once
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/octree/octree.h>
#include <pcl/point_cloud.h>
#include <pcl/common/centroid.h>
#include <pcl/features/boundary.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <omp.h>
#include <vector>
#include <Eigen/Dense>
#include "V3.hpp"
#ifndef PointType
#define PointType pcl::PointXYZRGBNormal
#endif
using namespace std;
double ComputeMeanDistance(const pcl::PointCloud<PointType>::ConstPtr cloud);
double ComputeMaxDistance(const pcl::PointCloud<PointType>::ConstPtr cloud);
PointType ComputeCentroid(const pcl::PointCloud<PointType>::ConstPtr cloud);
void TransformPointCloud(pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud_tf,Eigen::Affine3f tf);
vector<double> StatisticNearestDistance(const pcl::PointCloud<PointType>::ConstPtr cloud);

class EvalAndEvec
{
    public:
        double eigenvalue_[3];
        V3 eigenvector_[3];
        void GetEvalAndEvec(pcl::PointCloud<PointType>::Ptr cloud);
        EvalAndEvec(pcl::PointCloud<PointType>::Ptr cloud){
            GetEvalAndEvec(cloud);
        }
        EvalAndEvec(){};
};