#include "Statistics.h"
Statistics::Statistics(vector<float> dat)
{
	dat_=dat;		
	float sum=0;
	float min=INT_MAX;
	float max=-INT_MAX;
	
	// statistics
	for(int i=0;i<dat.size();i++)
	{
		sum+=dat[i];
		min=min<dat[i] ? min:dat[i];
		max=max>dat[i] ? max:dat[i];
	}
	sum_=sum;
	min_=min;
	max_=max;
	mean_=sum_/dat.size();
	
	sum=0;
	for(int i=0;i<dat.size();i++)
	{
		sum+=pow(dat[i]-mean_,2);				
	}
	stdevp_=sqrt(sum/dat.size());
	stdev_=sqrt(sum/(dat.size()-1));
}

double GaussErrorFunction(double x)
{	
	double denominator=1+0.278393*x+0.230389*x*x+0.000972*pow(x,3)+0.078108*pow(x,4);
	// double rst=1-1.0/pow(denominator,4);
	double rst=1-1.0/denominator;
	return rst;
}

double LossFunc(double x)
{
	return max((double)0.0f,GaussErrorFunction(x/sqrt(2)));
}