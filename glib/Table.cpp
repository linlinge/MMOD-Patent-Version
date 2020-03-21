#include "Table.h"
void Table::Resize(int n)
{
    records_.resize(n);
}

int Table::Rows()
{
    return records_.size();
}

void Table::EnableActive()
{
    flag_is_active_=1;
}

void Table::DisableActive()
{
    flag_is_active_=0;
}

void Table::GetMeanAndVariance(int item_index)
{    
    if(1==item_index){
        if(flag_is_active_==0){
            double sum=0;
            // calculate average
            for(int i=0;i<records_.size();i++){
                sum+=records_[i].item1_;
            }
            mean_=sum/records_.size();

            // calculate variance
            sum=0;
            for(int i=0;i<records_.size();i++){
                sum+=pow(records_[i].item1_-mean_,2);
            }
            sigma_=sqrt(sum/(records_.size()-1));
        }
        else if(flag_is_active_==1){
            double sum=0;
            // calculate average
            for(int i=0;i<unbelievable_idx_.size();i++){
                sum+=records_[unbelievable_idx_[i]].item1_;
            }
            mean_=sum/unbelievable_idx_.size();

            // calculate variance
            sum=0;
            for(int i=0;i<unbelievable_idx_.size();i++){
                sum+=pow(records_[unbelievable_idx_[i]].item1_-mean_,2);
            }
            sigma_=sqrt(sum/(unbelievable_idx_.size()-1));
        }
        else 
            cout<<"Error 01: active mode error!"<<endl;
    }
}

void Table::GetMinimumAndMaximum(int item_index)
{
    if(1==item_index){
        if(flag_is_active_==0){
            min_=INT_MAX;
            max_=-INT_MAX;
            // #pragma omp parallel for
            for(int i=0;i<records_.size();i++){
                min_=min_<records_[i].item1_ ? min_:records_[i].item1_;
                max_=max_>records_[i].item1_ ? max_:records_[i].item1_;
            }
        }
        else if(flag_is_active_==1){
            min_=INT_MAX;
            max_=-INT_MAX;
            for(int i=0;i<unbelievable_idx_.size();i++){
                min_=min_<records_[unbelievable_idx_[i]].item1_ ? min_:records_[unbelievable_idx_[i]].item1_;
                max_=max_>records_[unbelievable_idx_[i]].item1_ ? max_:records_[unbelievable_idx_[i]].item1_;
            }
        }
        else
            cout<<"Error 02: GetMinimumAndMaximum error!"<<endl;
    }    
}

void Table::Standardize_Zscore(int item_index)
{
    if(1==item_index){
        if(flag_is_active_==0){
            GetMeanAndVariance();
            #pragma omp parallel for
            for(int i=0;i<records_.size();i++){
                records_[i].item1_=(records_[i].item1_-mean_)/sigma_;
            }
        }
        else if(flag_is_active_==1){
            GetMeanAndVariance();
            #pragma omp parallel for
            for(int i=0;i<unbelievable_idx_.size();i++){
                records_[unbelievable_idx_[i]].item1_=(records_[unbelievable_idx_[i]].item1_-mean_)/sigma_;
            }
        }
        else
            cout<<"Error 03: Standardize Zscore error!"<<endl;        
    }
}

void Table::Normalize_Min_Max(int item_index)
{
    if(1==item_index){
        if(flag_is_active_==0){
            GetMinimumAndMaximum();
            #pragma omp parallel for
            for(int i=0;i<records_.size();i++)
                records_[i].item1_=(records_[i].item1_-min_)/(max_-min_);        
        }
        else if(flag_is_active_==1){
            GetMinimumAndMaximum();
            #pragma omp parallel for
            for(int i=0;i<unbelievable_idx_.size();i++)
                records_[unbelievable_idx_[i]].item1_=(records_[unbelievable_idx_[i]].item1_-min_)/(max_-min_); 
        }
        else
            cout<<"Error 04: Normalize min max error!"<<endl;
    }
}

void Table::Normalize_Tanh(double scale,int item_index)
{
    if(1==item_index){        
        if(flag_is_active_==0){
            GetMeanAndVariance();

            #pragma omp parallel for
            for(int i=0;i<records_.size();i++){
                records_[i].item1_=tanh(scale*records_[i].item1_);
            }    
        }
        else if(flag_is_active_==1){
            GetMeanAndVariance();

            #pragma omp parallel for
            for(int i=0;i<unbelievable_idx_.size();i++){
                records_[unbelievable_idx_[i]].item1_=tanh(scale*records_[unbelievable_idx_[i]].item1_);
            }
        }
        else
           cout<<"Error 05: Normalize Tanh error!"<<endl; 
    }
}

void Table::Ascending(int item_index)
{
    if(1==item_index)
        sort(records_.begin(),records_.end(),[](Rrd& e1,Rrd& e2){ return e1.item1_<e2.item1_;});
}

void Table::Descending(int item_index)
{
    if(1==item_index)
        sort(records_.begin(),records_.end(),[](Rrd& e1, Rrd& e2){ return e1.item1_> e2.item1_;});
}

void Table::Write(string path)
{
    ofstream fout(path);
    if(flag_is_active_==0){
        for(int i=0;i<records_.size();i++){
            fout<<records_[i].item1_<<endl;
        }
    }
    else if(flag_is_active_==1){
        GetBoxplot();
        for(int i=0;i<unbelievable_idx_.size();i++){
            fout<<records_[unbelievable_idx_[i]].item1_<<endl;
        }
    }
    fout.close();
}

void Table::MultiplyQ6(int m1,int m2,double scale)
{
    GetMinimumAndMaximum();
    double step=(max_-min_)/6.0;
    double lower_bound=(m1-1)*step+min_;
    double upper_bound=m2*step+min_;
    #pragma omp parallel for
    for(int i=0;i<records_.size();i++){
        if(records_[i].item1_>= lower_bound && records_[i].item1_<= upper_bound){
            records_[i].item1_=records_[i].item1_*scale;
        }
    }
}

void Table::Overturn()
{
    GetMinimumAndMaximum();
    // #pragma omp parallel for
    for(int i=0;i<records_.size();i++){        
        records_[i].item1_=max_-records_[i].item1_;        
    }
}

void Table::SetQ6(int m1,int m2)
{
    GetMinimumAndMaximum();
    double step=(max_-min_)/6.0;
    double lower_bound=(m1-1)*step+min_;
    double upper_bound=m2*step+min_;
    #pragma omp parallel for
    for(int i=0;i<records_.size();i++){
        if(records_[i].item1_>= lower_bound && records_[i].item1_<= upper_bound){
            records_[i].item1_=1.0f;
        }
    }
}

double Table::erf(double x)
{
	double a1=0.278393;
	double a2=0.230389;
	double a3=0.000972;
	double a4=0.078108;
	double m=1+a1*x+a2*pow(x,2)+a3*pow(x,3)+a4*pow(x,4);
	return (1-1.0/pow(m,4));
}

void Table::GetNormalDistributionError()
{
    #pragma omp parallel for
    for(int i=0;i<records_.size();i++){
        records_[i].item1_=erf(records_[i].item1_);
    }
}

void Table::LocalFilter(pcl::PointCloud<PointType>::Ptr cloud,int K)
{
    vector<double> lf;
    lf.resize(cloud->points.size());
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);

    // #pragma omp parallel for
    for(int i=0;i<cloud->points.size();i++){
        vector<int> idx(K+1);
        vector<float> dist(K+1);        
        kdtree->nearestKSearch(cloud->points[i], K+1, idx, dist);
        double sum=0;
        for(int j=1;j<K+1;j++){
            sum+=records_[idx[j]].item1_;
        }
        sum/=K;
        lf[i]=records_[i].item1_/sum-1.0f;
    }

    for(int i=0;i<cloud->points.size();i++){
        records_[i].item1_=lf[i];
    }
}

void Table::GetHistogram(int k)
{
    records_hist_.resize(k);
    for(int i=0;i<records_.size();i++){
        records_hist_[records_[i].item1_]+=1;
    }
}
void Table::GaussianOutlier(int k)
{
    GetMeanAndVariance();
    double lower_thresh=max((double)0.0,mean_-k*sigma_);
    double upper_thresh=mean_+2*sigma_;
    for(int i=0;i<records_.size();i++){
        if(records_[i].item1_<=lower_thresh){
            unbelievable_idx_.push_back(records_[i].id_);
        }

        if(records_[i].item1_>=upper_thresh){
            believable_idx_.push_back(records_[i].id_);
        }
    }
}

double Table::Quantile(double p)
{   
    SortBackup(1);
    double Q_idx=1+(records_sorted_.size()-1)*p;
    int Q_idx_integer=(int)Q_idx;
    double Q_idx_decimal=Q_idx-Q_idx_integer;
    double Q=records_sorted_[Q_idx_integer-1].item1_+(records_sorted_[Q_idx_integer].item1_-records_sorted_[Q_idx_integer-1].item1_)*Q_idx_decimal;    
    return Q;
}

double Table::Median()
{
    SortBackup(1);
    double median=0;
    if(records_sorted_.size()%2==1){// odd
        median= records_sorted_[(records_sorted_.size()+1.0)/2.0-1].item1_;
    }
    else{ // even        
        median=(records_sorted_[records_sorted_.size()/2-1].item1_+records_sorted_[records_sorted_.size()/2].item1_)/2.0f;
    }
    median_=median;
    return median;
}

void Table::SortBackup(int item_index)
{
    // Step 1: Does it have backup ?
    if(records_sorted_.size()==0){
        records_sorted_.resize(records_.size());
        for(int i=0;i<records_.size();i++){
            records_sorted_[i].id_=records_[i].id_;
            records_sorted_[i].item1_=records_[i].item1_;
        }
    }

    // Step 2: Sort Backup Records
    if(item_index==1){
        sort(records_sorted_.begin(),records_sorted_.end(),[](Rrd& e1, Rrd& e2){ return e1.item1_<e2.item1_;});
    }
    else if(item_index==0){
        sort(records_sorted_.begin(),records_sorted_.end(),[](Rrd& e1, Rrd& e2){ return e1.id_<e2.id_;});
    }
}

void Table::GetBoxplot(double lamda, int item_index)
{
    if(item_index==1){
        GetMinimumAndMaximum();
        GetMeanAndVariance();

        // Get Median
        median_=Median();
        // Q1
        Q1_=Quantile(0.25);
        // Q3
        Q3_=Quantile(0.75);
        // IQR
        double IQR=Q3_-Q1_;

        // Part 01: upper inner limit
        upper_inner_limit_=Q3_+lamda*IQR;
        int i=0;
        for(;i<records_sorted_.size();i++){
            if(records_sorted_[i].item1_>upper_inner_limit_){
                upper_inner_limit_=records_sorted_[i-1].item1_;
                break;
            }
        }
        if(i==records_sorted_.size())
            upper_inner_limit_=min(upper_inner_limit_,records_sorted_[i-1].item1_);

        // Part 02: lower inner limit
        lower_inner_limit_=Q1_-lamda*IQR;
        i=0;
        for(;i<records_sorted_.size();i++){
            if(records_sorted_[i].item1_>lower_inner_limit_){
                lower_inner_limit_=records_sorted_[i].item1_;
                break;
            }
        }

        if(i==0)
            lower_inner_limit_=max(lower_inner_limit_,records_sorted_[0].item1_);          
    }

    // Get all inactive index
    for(int i=0;i<records_.size();i++){
        if(records_[i].item1_< lower_inner_limit_){
            lower_unbelievable_idx_.push_back(i);
            unbelievable_idx_.push_back(i);
        }
        else if(records_[i].item1_ > upper_inner_limit_){
            upper_unbelievable_idx_.push_back(i);
            unbelievable_idx_.push_back(i);
        }
        else{
            believable_idx_.push_back(i);
        }
    }
}

void Table::push_back(Rrd e)
{
    records_.push_back(e);
}

void Table::Print()
{
    for(int i=0;i<records_.size();i++){
        cout<<records_[i].item1_<<" ";
    }
    cout<<endl;
}

void Table::GetCorrespondingColor(int item_index)
{
    if(item_index==1){
        color_.resize(records_.size());

        if(flag_is_active_==1){
            // Get Minimum and Maximum
            GetMinimumAndMaximum();

            // Lower inactive value
            for(int i=0;i<lower_unbelievable_idx_.size();i++){
                int itmp=lower_unbelievable_idx_[i];
                color_[itmp].r=0;
                color_[itmp].g=0;
                color_[itmp].b=255;
            }

            // Upper inactive value
            for(int i=0;i<upper_unbelievable_idx_.size();i++){
                int itmp=upper_unbelievable_idx_[i];
                color_[itmp].r=255;
                color_[itmp].g=0;
                color_[itmp].b=0;
            }

            // 
            for(int i=0;i<unbelievable_idx_.size();i++){
                int itmp=unbelievable_idx_[i];
                V3 ctmp=get_color(min_,max_,records_[itmp].item1_);                  
                color_[itmp].r=ctmp.r;
                color_[itmp].g=ctmp.g;
                color_[itmp].b=ctmp.b;
            }
        }
        else if(flag_is_active_==0){
            GetMinimumAndMaximum();

            for(int i=0;i<records_.size();i++){
                V3 ctmp=get_color(min_,max_,records_[i].item1_);
                color_[i].r=ctmp.r;
                color_[i].g=ctmp.g;
                color_[i].b=ctmp.b;

                // if(records_[i].item1_<27){
                //     color_[i].r=255;
                //     color_[i].g=0;
                //     color_[i].b=0;
                // }
            }
        }
        else{
            cout<<"Error: flag status error!"<<endl;
        }
    }
}

void Table::nPLOF(pcl::PointCloud<PointType>::Ptr cloud,int K)
{
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);
    
    vector<double> plof;    
    plof.resize(cloud->points.size());
	// #pragma omp parallel for
	for (int i = 0; i < cloud->points.size(); i++){        
        vector<int> idx(K+1);
		vector<float> dist(K+1);
		kdtree->nearestKSearch (cloud->points[i], K+1, idx, dist);
        double sum = 0;
        for (int j = 1; j < K+1; j++)
          sum += records_[idx[j]].item1_;
        sum /= K;
        plof[i] = records_[i].item1_ / sum  - 1.0f;
    }

    for(int i=0;i<cloud->points.size();i++){
        records_[i].item1_=plof[i];
    }
}

// blue -> green -> yellow -> red
void Table::SetActiveIndex(string erea_color, string status)
{
    GetMinimumAndMaximum();
    double n=4.0;
    double step=(max_-min_)/n;
    double Q1=min_+step;
    double Q2=min_+2*step;
    double Q3=min_+3*step;
    if(erea_color=="1000" && status == "believable"){
        for(int i=0;i<records_.size();i++){
            if(records_[i].item1_<=Q1){
                believable_idx_.push_back(i);
            }
            else{
                unbelievable_idx_.push_back(i);
            }
        }
    }
    else if(erea_color=="1100" && status == "believable"){
        for(int i=0;i<records_.size();i++){
            if(records_[i].item1_<=Q2){
                believable_idx_.push_back(i);
            }
            else{
                unbelievable_idx_.push_back(i);
            }
        }
    }
    else if(erea_color=="1110" && status == "believable"){
        for(int i=0;i<records_.size();i++){
            if(records_[i].item1_<=Q3){
                believable_idx_.push_back(i);
            }
            else{
                unbelievable_idx_.push_back(i);
            }
        }
    }
     
}

void Table::KDE()
{
    
}