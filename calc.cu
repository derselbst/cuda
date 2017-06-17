
// nvcc -std=c++11 calc.cu -o calc -res-usage -g -O2 -Xcompiler "-fopenmp"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include "types.h"

using namespace std;
using namespace std::chrono;

#define ACCESS(ARRAY, SET_IDX, LDA, ELEMENT) ARRAY[(SET_IDX) + (ELEMENT)*(LDA)]

__device__ __host__ bool fitPointsClobbered(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, real_t& slope_out, real_t& y_out);
__device__ __host__ bool calcContactPointClobbered(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, my_size_t& idx_out);

__device__ __host__ bool calcContactPointSoa(const point_alt_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, my_size_t& idx_out);
__device__ __host__ bool fitPointsSoa(const point_alt_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, real_t& slope_out, real_t& y_out);

__global__ void kernelClobbered(const point_t* pts, const my_size_t nSets, my_size_t* cuda_contacts, real_t* cuda_slopes, real_t* cuda_yIntersects)
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
    cuda_contacts[myAddr] = contactIdx;
    
    // polyfit sample data (first part)
    real_t slope=0;
    real_t yIntersect=0;
    fitPointsClobbered(&ACCESS(pts, 0, nSets, 2), contactIdx+1, // polyfit from 2 element (i.e. first data point) up to contact idx
              myAddr, nSets, slope, yIntersect);
              
              
    cuda_slopes[myAddr] = slope;
    cuda_yIntersects[myAddr] = yIntersect;
}


/*
__device__ ?? Calculate2LinearSegmentApprox(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, real_t& slope_out, real_t& y_out, my_size_t maxSizeFirstPart)
{
    my_size_t bestSplitIndex = -1;
    int currentQuality = 10000;
    cov1 = None
    cov2 = None
    dipSize = None
    for(my_size_t i=2; i<nPoints-3; i++)
    {
            if (i > maxSizeFirstPart):
                break
    
        // try to fit the first part from point 0 to i
        fitPointsClobbered(&ACCESS(pts, set_idx, lda, 0), i, myAddr, lda, slope, yIntersect);
        
        // try to fit the second part from point i to i
        fitPointsClobbered(&ACCESS(pts, set_idx, lda, i), nPoints-i, myAddr, lda, slope, yIntersect);
        

        lin1 = numpy.polyfit(firstPart[:, 0], firstPart[:, 1], 1, full=True)
        lin2 = numpy.polyfit(secondPart[:, 0], secondPart[:, 1], 1, full=True)

        res1 = 0
        if (len(lin1[1]) > 0):
            res1 = lin1[1][0]
        res2 = lin2[1][0]

        # minimize this
        # overalQuality = res1/ i + res2 / (nPoints -1);
        overalQuality = res1 + res2;

        if (overalQuality < currentQuality) or bestSplitIndex < 0:
            bestSplitIndex = i
            currentQuality = overalQuality

            cov1 = lin1[0]
            cov2 = lin2[0]
            dipSize = math.fabs(firstPart[0][1] - firstPart[i-1][1])

    if (cov1 is None):
        return None


    return numpy.concatenate((cov1,cov2)) , bestSplitIndex , dipSize;
}*/
    
__device__ __host__ bool fitPointsClobbered(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, real_t& slope_out, real_t& y_out)
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

__device__ __host__ bool calcContactPointClobbered(const point_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, my_size_t& idx_out)
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

__host__ void checkResultsClobbered(const vector<point_t>& sets_clobbered, const my_size_t* cuda_contacts, const real_t* cuda_slopes, const real_t* cuda_yIntersects, my_size_t nSets)
{
    #pragma omp parallel for schedule(static) 
    for(my_size_t myAddr=0; myAddr < nSets; myAddr++)
    {
        const my_size_t nPoints = ACCESS(sets_clobbered.data(), myAddr, nSets, 0).n;
    
        my_size_t contactIdx;
        // get contact idx and split idx
        bool succ = calcContactPointClobbered(&ACCESS(sets_clobbered.data(), 0, nSets, 2), nPoints, myAddr, nSets, contactIdx);
        
        if(succ && contactIdx != cuda_contacts[myAddr])
        {
            cerr << "contactIdx mismatch at set " << myAddr << ": expected " << cuda_contacts[myAddr] << "; actual " << contactIdx << endl;
        }

        // polyfit sample data (first part)
        real_t slope=0;
        real_t yIntersect=0;
        succ = fitPointsClobbered(&ACCESS(sets_clobbered.data(), 0, nSets, 2), contactIdx+1, // polyfit from 2 element (i.e. first data point) up to contact idx
                myAddr, nSets, slope, yIntersect);
                
//         if(succ)
        {
            if(slope != cuda_slopes[myAddr])
            {
                cerr << "slope mismatch at set " << myAddr << ": expected " << cuda_slopes[myAddr] << "; actual " << slope << endl;
            }
            
            if(yIntersect != cuda_yIntersects[myAddr])
            {
                cerr << "yIntersect mismatch at set " << myAddr << ": expected " << cuda_yIntersects[myAddr] << "; actual " << yIntersect << endl;
            }
        }
    }
}

__global__ void kernelSoa(const my_size_t* rowsPerThread, const point_alt_t* pts, const my_size_t nSets)
{
    int tid = threadIdx.x;    //lokaler Thread Index
    int bid = blockIdx.x;     //Index des Blockes
    int bdim= blockDim.x;     //Anzahl an Threads pro Block

    int myAddr = tid+bid*bdim;

    const my_size_t nPoints = rowsPerThread[myAddr];

    my_size_t contactIdx;
    // get contact idx and split idx
    calcContactPointSoa(&ACCESS(pts, 0, nSets, 0), nPoints, myAddr, nSets, contactIdx);

    // polyfit sample data (first part)
    real_t slope;
    real_t yIntersect;
    fitPointsSoa(&ACCESS(pts, 0, nSets, 0), contactIdx+1, // polyfit from 2 element (i.e. first data point) up to contact idx
               myAddr, nSets, slope, yIntersect);
}

__device__ __host__ bool calcContactPointSoa(const point_alt_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, my_size_t& idx_out)
{
    for (my_size_t i=1; i<nPoints; i++)
    {
        const point_alt_t prev= pts[i-1];
        const real_t prevZ = prev.z[set_idx];
        const real_t prevF = prev.force[set_idx];
        
        const point_alt_t cur = pts[i];
        const real_t curZ = cur.z[set_idx];
        const real_t curF = cur.force[set_idx];

        const real_t deltaZ     = curZ - prevZ;
        const real_t deltaForce = curF - prevF;

        const real_t avg = (curZ + prevZ)/2.0;
        const real_t slope = deltaForce / deltaZ;
        if (slope > 0)
        {
            idx_out = i;
            return true;
        }
    }
    return false;
}

__device__ __host__ bool fitPointsSoa(const point_alt_t* pts, my_size_t nPoints, const my_size_t set_idx, const my_size_t lda, real_t& slope_out, real_t& y_out)
{
    if(nPoints <= 1)
    {
        // Fail: infinitely many lines passing through this single point
        return false;
    }

    real_t sumX=0, sumY=0, sumXY=0, sumXX=0;
    for(my_size_t i=0; i<nPoints; i++)
    {
        const point_alt_t tmp = pts[i];
        const real_t curZ = tmp.z[set_idx];
        const real_t curF = tmp.force[set_idx];
        sumX += curZ;
        sumY += curF;
        sumXY += curZ * curF;
        sumXX += curZ * curZ;
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

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        cout << "usage: " << argv[0] << " BINARY_BLOB" << endl;
        return -1;
    }

    ifstream in(argv[1]);

    my_size_t columns; // number of sets
    my_size_t rows; // number of points in each set
    in.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    
    vector<point_t> sets_clobbered(columns*rows);
    in.read(reinterpret_cast<char*>(sets_clobbered.data()), sizeof(point_t) * columns * rows);
    
    
    const dim3 threads(1024);
    const dim3 grid(columns/threads.x);
    
    point_t* sets_clobbered_cuda = nullptr;
    my_size_t* cuda_contacts = nullptr;
    real_t* cuda_slopes = nullptr;
    real_t* cuda_yIntersects = nullptr;    
    if(cudaMalloc(&sets_clobbered_cuda, sizeof(*sets_clobbered_cuda) * columns*rows) != cudaSuccess) return -1;
    cudaMemcpy(sets_clobbered_cuda, sets_clobbered.data(), sizeof(*sets_clobbered_cuda) * columns*rows, cudaMemcpyHostToDevice);
    
    if(cudaMalloc(&cuda_contacts, sizeof(*cuda_contacts) * columns) != cudaSuccess) return -1;
    if(cudaMalloc(&cuda_slopes, sizeof(*cuda_slopes) * columns) != cudaSuccess) return -1;
    if(cudaMalloc(&cuda_yIntersects, sizeof(*cuda_yIntersects) * columns) != cudaSuccess) return -1;
   
    
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernelClobbered<<<grid, threads>>>(sets_clobbered_cuda, columns, cuda_contacts, cuda_slopes, cuda_yIntersects);
//    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaError_t err=cudaGetLastError();
    if (err!=cudaSuccess)
    {
        printf("An Error occured in clobbered kernel: %s (%i)\n",cudaGetErrorString(err),err);
        return(-1);
    }
    float kernelClobbered_time;
    cudaEventElapsedTime(&kernelClobbered_time, start, stop);

    cudaFree(sets_clobbered_cuda);
    sets_clobbered_cuda = nullptr;
    
    
    vector<my_size_t> contactResults(columns);
    cudaMemcpy(contactResults.data(), cuda_contacts, sizeof(*cuda_contacts) * columns, cudaMemcpyDeviceToHost);
    cudaFree(cuda_contacts);
    cuda_contacts = nullptr;
        
    vector<real_t> slopesResults(columns);
    cudaMemcpy(slopesResults.data(), cuda_slopes, sizeof(*cuda_slopes) * columns, cudaMemcpyDeviceToHost);
    cudaFree(cuda_slopes);
    cuda_slopes = nullptr;
    
    vector<real_t> yIntsctResults(columns);
    cudaMemcpy(yIntsctResults.data(), cuda_yIntersects, sizeof(*cuda_yIntersects) * columns, cudaMemcpyDeviceToHost);
    cudaFree(cuda_yIntersects);
    cuda_yIntersects = nullptr;
    
    duration<double, std::milli> cpuClobbered_time;
    {
    auto t1 = high_resolution_clock::now();
    checkResultsClobbered(sets_clobbered, contactResults.data(), slopesResults.data(), yIntsctResults.data(), columns);
    auto t2 = high_resolution_clock::now();
    cpuClobbered_time = t2 - t1;
    }
    
    /*** alternative attempt with pure SoA***/
    
    point_alt_t* soaPointsCuda = nullptr;
    vector<point_alt_t> soaPoints; // array of soa structs holding the data points for each thread
    
    vector<my_size_t> pointsPerSet(columns); // no. of points each thread processes
    for(my_size_t i=0; i<columns; i++)
    {
        pointsPerSet[i] = ACCESS(sets_clobbered.data(), i, columns, 0).n;
    }
    my_size_t* pointsPerSetCuda = nullptr;
    if(cudaMalloc(&pointsPerSetCuda, sizeof(*pointsPerSetCuda) * columns) != cudaSuccess) goto fail3;
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
    cudaMemcpy(soaPointsCuda, soaPoints.data(), sizeof(*soaPointsCuda) * rows, cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    kernelSoa<<<grid, threads>>>(pointsPerSetCuda, soaPointsCuda, columns);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
//    cudaDeviceSynchronize();
    err=cudaGetLastError();
    if (err!=cudaSuccess)
    {
        printf("An Error occured in soa kernel: %s (%i)\n",cudaGetErrorString(err),err);
        return(-1);
    }
    float kernelSoa_time;
    cudaEventElapsedTime(&kernelSoa_time, start, stop);
    
    
    cout << "gpu timing in ms:\n" << "  kernelClobbered: " << kernelClobbered_time << "\n  kernelSoa: " << kernelSoa_time << endl;
    cout << "cpu timing in ms:\n" << "  cpuClobbered: " << cpuClobbered_time.count() << "\n  cpuSoa: " << endl;
    
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
