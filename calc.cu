
#include "types.h"


bool fitPoints(const point_t* pts, const my_size_t set_idx, const my_size_t lda, real_t& slope_out, real_t& y_out)
{
#define ACCESS(ELEMENT) pts[set_idx + ELEMENT*lda]

    const my_size_t nPoints = ACCESS(0).n;
    if(nPoints <= 1)
    {
        // Fail: infinitely many lines passing through this single point
        return false;
    }
    real_t sumX=0, sumY=0, sumXY=0, sumXX=0;
    for(my_size_t i=0; i<nPoints; i++)
    {
        point_t tmp = ACCESS(i);
        sumX += tmp.z;
        sumY += tmp.force;
        sumXY += tmp.z * tmp.force;
        sumXX += tmp.z * tmp.z;
    }
    const real_t xMean = sumX / nPoints;
    const real_t yMean = sumY / nPoints;
    const real_t denominator = sumXX - sumX * xMean;
    
    if(std::fabs(denominator) < 1e-5f)
    {
        // seems a vertical line
        return false;
    }
    slope_out = (sumXY - sumX * yMean) / denominator;
    y_out = yMean - slope_out * xMean;
    return true;

#undef ACCESS
}

int main()
{

}