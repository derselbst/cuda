
// nvcc -std=c++11 calc.cu -o calc -res-usage -g

#include <iostream>
#include <fstream>
#include <vector>

#include "types.h"

using namespace std;

#define ACCESS(ARRAY, SET_IDX, LDA, ELEMENT) ARRAY[SET_IDX + ELEMENT*LDA]

__device__ bool fitPointsClobbered(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, real_t& slope_out, real_t& y_out);
__device__ bool calcContactPointClobbered(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, my_size_t& idx_out);

__global__ void kernelClobbered(const point_t* pts, const my_size_t nSets)
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
    calcContactPointClobbered(&ACCESS(pts, 0, nSets, 2), nPoints, myAddr, nSets, contactIdx);

    // polyfit sample data (first part)
    real_t slope;
    real_t yIntersect;
    fitPointsClobbered(&ACCESS(pts, 0, nSets, 2), contactIdx+1, // polyfit from 2 element (i.e. first data point) up to contact idx
              myAddr, nSets, slope, yIntersect);
}

__device__ bool fitPointsClobbered(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, real_t& slope_out, real_t& y_out)
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

__device__ bool calcContactPointClobbered(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, my_size_t& idx_out)
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

__global__ void kernelSoa(const my_size_t* rowsPerThread, const point_alt_t* pts, const my_size_t nSets)
{
    int tid = threadIdx.x;    //lokaler Thread Index
    int bid = blockIdx.x;     //Index des Blockes
    int bdim= blockDim.x;     //Anzahl an Threads pro Block

    int myAddr = tid+bid*bdim;

    const my_size_t nPoints = ACCESS(rowsPerThread, myAddr, nSets, 0);

    my_size_t contactIdx;
    // get contact idx and split idx
//     calcContactPoint(&ACCESS(pts, 0, nSets, 0), nPoints, myAddr, nSets, contactIdx);

    // polyfit sample data (first part)
    real_t slope;
    real_t yIntersect;
//     fitPointsd(&ACCESS(pts, 0, nSets, 0), contactIdx+1, // polyfit from 2 element (i.e. first data point) up to contact idx
//               myAddr, nSets, slope, yIntersect);
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        cout << "usage: " << argv[0] << " BINARY_BLOB" << endl;
        return -1;
    }

    ifstream in(argv[1]);

    my_size_t columns;
    my_size_t rows;
    in.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));

    vector<point_t> sets_clobbered(columns*rows);
    in.read(reinterpret_cast<char*>(sets_clobbered.data()), sizeof(point_t) * columns * rows);
    
    point_t* sets_clobbered_cuda = nullptr;
    if(cudaMalloc(&sets_clobbered_cuda, sizeof(*sets_clobbered_cuda) * columns*rows) != cudaSuccess) return -1;
   
    dim3 threads(1024);
    dim3 grid(columns/threads.x);
    
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernelClobbered<<<grid, threads>>>(sets_clobbered_cuda, columns);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernelClobbered_time;
    cudaEventElapsedTime(&kernelClobbered_time, start, stop);

    cudaFree(sets_clobbered_cuda);
    sets_clobbered_cuda = nullptr;
    
    /*** alternative attempt with pure SoA***/
    
    point_alt_t* soaPointsCuda = nullptr;
    vector<point_alt_t> soaPoints; // array of soa structs holding the data points for each thread
    
    vector<my_size_t> pointsPerSet(columns); // no. of points each thread processes
    my_size_t* pointsPerSetCuda = nullptr;
    if(cudaMalloc(&pointsPerSetCuda, sizeof(*pointsPerSetCuda) * columns) != cudaSuccess) goto fail3;
    for(my_size_t i=0; i<columns; i++)
    {
        pointsPerSet[i] = ACCESS(sets_clobbered.data(), i, columns, 0).n;
    }
    cudaMemcpy(pointsPerSetCuda, pointsPerSet.data(), sizeof(*pointsPerSetCuda) * columns, cudaMemcpyHostToDevice);

    rows--; // first row of sets_clobbered containing sizes, just read them
    rows--; // here are the x,y positions stored, which we ignore for now

    soaPoints.resize(rows);
    if(cudaMalloc(&soaPointsCuda, sizeof(*soaPointsCuda) * rows) != cudaSuccess) goto fail2;
    for(my_size_t i=0; i<rows; i++)
    {
        // alloc temp host mem to store to datapoints to
        soaPoints[i].z = new (nothrow) real_t[columns];
        soaPoints[i].force = new (nothrow) real_t[columns];
        
        if(soaPoints[i].force == nullptr || soaPoints[i].z == nullptr)
        {
            delete [] soaPoints[i].force;
            delete [] soaPoints[i].z;
            soaPoints[i].z = nullptr;
            soaPoints[i].force = nullptr;
            goto fail;
        }

        // store the data points for each set
        for(my_size_t j=0; j<columns; j++)
        {
            point_t tmp = ACCESS(sets_clobbered.data(), j, columns, i+2);
            soaPoints[i].z[j] = tmp.z;
            soaPoints[i].force[j] = tmp.force;
        }

        // alloc device mem for data points
        real_t* tmpCudaZ = nullptr;
        real_t* tmpCudaForce = nullptr;
        if(cudaMalloc(&tmpCudaZ, sizeof(real_t) * columns) != cudaSuccess) goto fail;
        if(cudaMalloc(&tmpCudaForce, sizeof(real_t) * columns) != cudaSuccess)
        {
            cudaFree(tmpCudaZ);
            delete [] soaPoints[i].force;
            delete [] soaPoints[i].z;
            soaPoints[i].z = nullptr;
            soaPoints[i].force = nullptr;
            goto fail;
        }
        
        // copy the datapoints from temp host mem to dev mem
        cudaMemcpy(tmpCudaZ, soaPoints[i].z, sizeof(real_t) * columns, cudaMemcpyHostToDevice);
        cudaMemcpy(tmpCudaForce, soaPoints[i].force, sizeof(real_t) * columns, cudaMemcpyHostToDevice);
        
        // free host mem, make pointers point to dev mem array
        delete [] soaPoints[i].z;
        soaPoints[i].z = tmpCudaZ;
        delete [] soaPoints[i].force;
        soaPoints[i].force = tmpCudaForce;
    }
    cudaMemcpy(soaPointsCuda, soaPoints.data(), sizeof(real_t) * columns, cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    kernelSoa<<<grid, threads>>>(pointsPerSetCuda, soaPointsCuda, columns);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernelSoa_time;
    cudaEventElapsedTime(&kernelSoa_time, start, stop);
    
    
    cout << "gpu timing in ms:\n" << "kernelClobbered: " << kernelClobbered_time << "\nkernelSoa: " << kernelSoa_time << endl;
    
fail:
    for(my_size_t i=0; i<rows; i++)
    {
        cudaFree(soaPoints[i].z);
        cudaFree(soaPoints[i].force);
    }
    cudaFree(soaPointsCuda);
    
fail2:
    cudaFree(pointsPerSetCuda);
    
fail3:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
