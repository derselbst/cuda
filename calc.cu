
#include "types.h"

#define ACCESS(ARRAY, SET_IDX, LDA, ELEMENT) ARRAY[SET_IDX + ELEMENT*lda]

__global__ void kernel()
{

    const my_size_t nPoints = ACCESS(pts, myAddr, nCols, 0).n;

    const float x = ACCESS(pts, myAddr, nCols, 1).z;
    const float y = ACCESS(pts, myAddr, nCols, 1).force;
    
    // get contact idx and split idx
    
    // polyfit sample data (first part)
    real_t slope;
    real_t yIntersect;
    fitPoints(&ACCESS(pts, 0, nCols, 2), contact_idx+1, // polyfit from 2 element (i.e. first data point) up to contact idx
              myAddr, nCols, slope, yIntersect);
}

bool fitPoints(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, real_t& slope_out, real_t& y_out)
{
    if(nPoints <= 1)
    {
        // Fail: infinitely many lines passing through this single point
        return false;
    }
    
    real_t sumX=0, sumY=0, sumXY=0, sumXX=0;
    for(my_size_t i=0; i<nPoints; i++)
    {
        point_t tmp = ACCESS(pts, set_idx, lda, i);
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
}

int main()
{

}
