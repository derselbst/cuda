
#include "types.h"

#define ACCESS(ARRAY, SET_IDX, LDA, ELEMENT) ARRAY[SET_IDX + ELEMENT*lda]

__global__ void kernel(const point_t* pts, const my_size_t nSets)
{

    const my_size_t nPoints = ACCESS(pts, myAddr, nSets, 0).n;

    const float x = ACCESS(pts, myAddr, nSets, 1).z;
    const float y = ACCESS(pts, myAddr, nSets, 1).force;
    
    // get contact idx and split idx
    
    // polyfit sample data (first part)
    real_t slope;
    real_t yIntersect;
    fitPoints(&ACCESS(pts, 0, nSets, 2), contact_idx+1, // polyfit from 2 element (i.e. first data point) up to contact idx
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
        const point_t tmp = ACCESS(pts, set_idx, lda, i);
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

bool calcContactPoint(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, my_size_t& idx_out)
{
    for (my_size_t i=1; i<nPoints; i++)
    {
        const point_t cur = ACCESS(pts, set_idx, lda, i);
        const point_t prev= ACCESS(pts, set_idx, lda, i-1);
        
        const real_t deltaZ     = cur.z - prev.z;
        const real_t deltaForce = cur.force - prev.force;
        
        const real_t avg = (cur.z + prev.z)/2.0;
        const real_t slope = deltaForce / deltaZ;
        if (slope > 0)
        {
            idx_out = i;
            return true;
        }
    }
    return false;
}

int main()
{

}
