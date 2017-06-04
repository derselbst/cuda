#include <iostream>
#include <fstream>
#include <vector>

#include "types.h"

using namespace std;

#define ACCESS(ARRAY, SET_IDX, LDA, ELEMENT) ARRAY[SET_IDX + ELEMENT*LDA]

__device__ bool fitPoints(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, real_t& slope_out, real_t& y_out);
__device__ bool calcContactPoint(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, my_size_t& idx_out);

__global__ void kernel(const point_t* pts, const my_size_t nSets)
{
    int tid = threadIdx.x;    //lokaler Thread Index
    int bid = blockIdx.x;     //Index des Blockes
    int bdim= blockDim.x;     //Anzahl an Threads pro Block

    int myAddr = tid+bid*bdim;

    const my_size_t nPoints = ACCESS(pts, myAddr, nSets, 0).n;

    const float x = ACCESS(pts, myAddr, nSets, 1).z;
    const float y = ACCESS(pts, myAddr, nSets, 1).force;

    my_size_t contactIdx;
    // get contact idx and split idx
    calcContactPoint(&ACCESS(pts, 0, nSets, 2), nPoints, myAddr, nSets, contactIdx);

    // polyfit sample data (first part)
    real_t slope;
    real_t yIntersect;
    fitPoints(&ACCESS(pts, 0, nSets, 2), contactIdx+1, // polyfit from 2 element (i.e. first data point) up to contact idx
              myAddr, nSets, slope, yIntersect);
}

__device__ bool fitPoints(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, real_t& slope_out, real_t& y_out)
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

__device__ bool calcContactPoint(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, my_size_t& idx_out)
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

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        cout << "usage: " << argv[0] << " BINARY_BLOB" << endl;
        return -1;
    }

    ifstream in(argv[1]);

    size_t columns;
    size_t rows;
    in.read(reinterpret_cast<char*>(&columns), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));

    vector<point_t> sets_clobbered(columns*rows);
    in.read(reinterpret_cast<char*>(sets_clobbered.data()), sizeof(point_t) * columns * rows);
   
    dim3 threads(1024);
    dim3 grid(columns/threads.x);
    kernelClobbered<<<grid, threads>>>(sets_clobbered_cuda, columns);

    my_size_t* pointsPerSetCuda = nullptr;
    if(cudaMalloc(&pointsPerSetCuda, sizeof(*pointsPerSetCuda) * columns) != cudaSuccess) return -1;
    vector<my_size_t> pointsPerSet(columns);
    for(my_size_t i=0; i<columns; i++)
    {
        pointsPerSet[i] = ACCESS(sets_clobbered.data(), i, columns, 0).n;
    }
    cudaMemcpy(pointsPerSetCuda, pointsPerSet.data(), sizeof(my_size_t) * columns, cudaMemcpyHostToDevice);


    vector<point_alt_t> soaPoints(rows);
    point_alt_t* soaPointsCuda = nullptr;
    if(cudaMalloc(&soaPointsCuda, sizeof(*soaPointsCuda) * rows) != cudaSuccess) goto fail2;
    for(my_size_t i=0; i<rows; i++)
    {
        soaPoints[i].z = new real_t[columns];
        soaPoints[i].force = new real_t[columns];

        if(cudaMalloc(&soaPointsCuda[i].z, sizeof(real_t) * columns) != cudaSuccess) goto fail;
        if(cudaMalloc(&soaPointsCuda[i].force, sizeof(real_t) * columns) != cudaSuccess) goto fail;

        for(my_size_t j=0; j<columns; j++)
        {
            point_t tmp = ACCESS(sets_clobbered.data(), j, columns, i+2);
            soaPoints[i].z[j] = tmp.z;
            soaPoints[i].force[j] = tmp.force;
        }

        cudaMemcpy(soaPointsCuda[i].z, soaPoints[i].z, sizeof(real_t) * columns, cudaMemcpyHostToDevice);
        cudaMemcpy(soaPointsCuda[i].force, soaPoints[i].force, sizeof(real_t) * columns, cudaMemcpyHostToDevice);
    }
    
    
    
fail:
    for(my_size_t i=0; i<rows; i++)
    {
        delete [] soaPoints[i].z;
        delete [] soaPoints[i].force;
    }

    if(soaPointsCuda != nullptr)
    {
        for(my_size_t i=0; i<rows; i++)
        {
            cudaFree(soaPointsCuda[i].z);
            cudaFree(soaPointsCuda[i].force);
        }
    }
fail2:    
    cudaFree(soaPointsCuda);

    cudaFree(pointsPerSetCuda);

}
