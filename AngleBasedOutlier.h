#pragma once
#include <math.h>
#include <vector>
#include "PCLExtend.h"
#include "Table.h"
using namespace std;
class RPY
{
    public:
        int id_;
        double x_,y_,z_;
        double roll_,pitch_,yaw_;
        double feature_;

        RPY(int id, double x, double y, double z){
            id_=id;
            double r=sqrt(pow(x,2)+pow(y,2)+pow(z,2));
            
            // X axis
            roll_=atan2(z,y);
            if(roll_<M_PI) roll_+=M_PI;

            // Y axis
            pitch_=atan2(x,z);
            if(pitch_<M_PI)  pitch_+=M_PI;

            // Z axis
            yaw_=atan2(y,x);
            if(yaw_<M_PI) yaw_+=M_PI;

            // Calculate Feature
            feature_=roll_+pitch_+yaw_;
        }
};

// One Point Entropy
class AngleEntropy
{
    public:
        vector<int> cells_;
        double dtheta,dphi;
        int NumOfCells_;
        double entropy_;
        int cols_,rows_;
        
        // Internal Function


        // External Function
        void Init(int rows,int cols){            
            rows_=rows;
            cols_=cols;

            cells_.resize(rows*cols);
            dtheta=2*M_PI/cols;
            dphi=M_PI/rows;
            NumOfCells_=rows*cols;
        }

        void Insert(double x, double y, double z){
            double r=sqrt(pow(x,2)+pow(y,2)+pow(z,2));
            double theta=acos(z/r);
            double phi=atan2(y,x)+M_PI;

            int i=floor(theta/dtheta);
            int j=floor(phi/dphi);
            cells_[i*cols_+j]+=1;
        }

        double GetEntropy()
        {
            double sum=0;
            for(int i=0;i<rows_;i++){
                for(int j=0;j<cols_;j++){
                    double prob=cells_[i*cols_+j]*1.0f/NumOfCells_;
                    if(prob>0.00001)
                        sum+=prob*log(prob);
                }
            }
            entropy_=-1.0f*sum;
            return entropy_;
        }
        ~AngleEntropy(){
            cells_.clear();
        }
};

// many points entropy
class AngleBasedOutlier
{
    public:       
       vector<Rrd> point_set_entropy_;
       pcl::PointCloud<PointType>::Ptr cloud_;
       
       // Points' Entropy
       void PointSetEntropy(pcl::PointCloud<PointType>::Ptr cloud);
       // Distance of centroid and centre
       void PointSetCCD(pcl::PointCloud<PointType>::Ptr cloud);
};