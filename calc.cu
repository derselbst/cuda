
// nvcc -std=c++11 calc.cu -o calc -res-usage -arch=compute_61 -g -O2 -Xcompiler "-fopenmp"

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

    if(myAddr < nSets)
    {
//         for(my_size_t k=0; k<1000; k++)
        {
            const my_size_t nPoints = ACCESS(pts, myAddr, nSets, 0).n;

            const float x = ACCESS(pts, myAddr, nSets, 1).z;
            const float y = ACCESS(pts, myAddr, nSets, 1).force;

            my_size_t contactIdx=0;
            // get contact idx and split idx
            calcContactPointClobbered(&ACCESS(pts, 0, nSets, 2), nPoints, myAddr, nSets, contactIdx);
            __syncthreads();
            
            cuda_contacts[myAddr] = contactIdx;
            
            // polyfit sample data (first part)
            real_t slope=0;
            real_t yIntersect=0;
            fitPointsClobbered(&ACCESS(pts, 0, nSets, 2), contactIdx+1, // polyfit from 2 element (i.e. first data point) up to contact idx
                    myAddr, nSets, slope, yIntersect);
            __syncthreads();
                    
            cuda_slopes[myAddr] = slope;
            cuda_yIntersects[myAddr] = yIntersect;
            
            // fit from contact point to split index (guessed)
            my_size_t splitIdx = contactIdx+10;
            fitPointsClobbered(&ACCESS(pts, 0, nSets, 2+(contactIdx+1)), (splitIdx-contactIdx), myAddr, nSets, slope, yIntersect);
            
            // fit from split index to end
            fitPointsClobbered(&ACCESS(pts, 0, nSets, 2+splitIdx), nPoints-splitIdx-1, myAddr, nSets, slope, yIntersect);
        }
    }
}
    
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

    if(std::fabs(denominator) < 1e-7f)
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
    const point_t* pts = sets_clobbered.data();
    
    #pragma omp parallel for schedule(dynamic) default(none) shared(cerr) firstprivate(pts, cuda_contacts, cuda_slopes, nSets, cuda_yIntersects)
    for(my_size_t myAddr=0; myAddr < nSets; myAddr++)
    {
        const my_size_t nPoints = ACCESS(pts, myAddr, nSets, 0).n;
    
        my_size_t contactIdx;
        // get contact idx and split idx
        bool succ = calcContactPointClobbered(&ACCESS(pts, 0, nSets, 2), nPoints, myAddr, nSets, contactIdx);
        
        if(succ && contactIdx != cuda_contacts[myAddr])
        {
//             cerr << "contactIdx mismatch at set " << myAddr << ": expected " << cuda_contacts[myAddr] << "; actual " << contactIdx << endl;
        }

        // polyfit sample data (first part)
        real_t slope=0;
        real_t yIntersect=0;
        succ = fitPointsClobbered(&ACCESS(pts, 0, nSets, 2), contactIdx+1, // polyfit from 2 element (i.e. first data point) up to contact idx
                myAddr, nSets, slope, yIntersect);
                
        if(succ)
        {
            if(slope != cuda_slopes[myAddr])
            {
//                 cerr << "slope mismatch at set " << myAddr << ": expected " << cuda_slopes[myAddr] << "; actual " << slope << endl;
            }
            
            if(yIntersect != cuda_yIntersects[myAddr])
            {
//                 cerr << "yIntersect mismatch at set " << myAddr << ": expected " << cuda_yIntersects[myAddr] << "; actual " << yIntersect << endl;
            }
        }
        
        fitPointsClobbered(&ACCESS(pts, 0, nSets, 2+contactIdx+1), nPoints-contactIdx-1, myAddr, nSets, slope, yIntersect);
    }
}

__global__ void kernelSoa(const my_size_t* rowsPerThread, const point_alt_t* pts, const my_size_t nSets, my_size_t* cuda_contacts, real_t* cuda_slopes, real_t* cuda_yIntersects)
{
    int tid = threadIdx.x;    //lokaler Thread Index
    int bid = blockIdx.x;     //Index des Blockes
    int bdim= blockDim.x;     //Anzahl an Threads pro Block

    int myAddr = tid+bid*bdim;

    if(myAddr < nSets)
    {
//         for(my_size_t k=0; k<1000; k++)
        {
            const my_size_t nPoints = rowsPerThread[myAddr];

            my_size_t contactIdx=0;
            // get contact idx and split idx
            calcContactPointSoa(&pts[0], nPoints, myAddr, nSets, contactIdx);
            __syncthreads();
            
            cuda_contacts[myAddr] = contactIdx;

            // polyfit sample data (first part)
            real_t slope=0;
            real_t yIntersect=0;
            fitPointsSoa(&pts[0], contactIdx+1, myAddr, nSets, slope, yIntersect);
            __syncthreads();
            
            cuda_slopes[myAddr] = slope;
            cuda_yIntersects[myAddr] = yIntersect;
            
            // guess split index
            my_size_t splitIndex = contactIdx+10;
            // polyfit from contactidx to splitidx
            fitPointsSoa(&pts[contactIdx+1], (splitIndex-contactIdx), myAddr, nSets, slope, yIntersect);
            
            //polyfit from split idx to end
            fitPointsSoa(&pts[splitIndex+1], nPoints-splitIndex-1, myAddr, nSets, slope, yIntersect);
        }
    }
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

__host__ void checkResultsSoa(const my_size_t* cuda_contactsSoa, const real_t* cuda_slopesSoa, const real_t* cuda_yIntersectsSoa, const my_size_t* cuda_contacts, const real_t* cuda_slopes, const real_t* cuda_yIntersects, const my_size_t nSets)
{
    #pragma omp parallel for schedule(dynamic) default(none) shared(cerr) firstprivate(cuda_contactsSoa, cuda_contacts, cuda_slopesSoa, cuda_slopes, nSets, cuda_yIntersectsSoa, cuda_yIntersects)
    for(my_size_t myAddr=0; myAddr < nSets; myAddr++)
    {
        if(cuda_contactsSoa[myAddr] != cuda_contacts[myAddr])
        {
            cerr << "contactIdx mismatch at set " << myAddr << ": expected " << cuda_contacts[myAddr] << "; actual " << cuda_contactsSoa[myAddr] << endl;
        }
        
        if(cuda_slopesSoa[myAddr] != cuda_slopes[myAddr])
        {
            cerr << "slope mismatch at set " << myAddr << ": expected " << cuda_slopes[myAddr] << "; actual " << cuda_slopesSoa[myAddr] << endl;
        }
        
        if(cuda_yIntersectsSoa[myAddr] != cuda_yIntersects[myAddr])
        {
            cerr << "yIntersect mismatch at set " << myAddr << ": expected " << cuda_yIntersects[myAddr] << "; actual " << cuda_yIntersectsSoa[myAddr] << endl;
        }
    }
}

__host__ void calcCpuSoa(const my_size_t* rowsPerThread, const point_alt_t* pts, const my_size_t nSets)
{
    #pragma omp parallel for schedule(dynamic) default(none) firstprivate(rowsPerThread,pts,nSets) 
    for(my_size_t myAddr=0; myAddr < nSets; myAddr++)
    {
        const my_size_t nPoints = rowsPerThread[myAddr];

        my_size_t contactIdx=0;
        // get contact idx and split idx
        calcContactPointSoa(&pts[0], nPoints, myAddr, nSets, contactIdx);

        // polyfit sample data (first part)
        real_t slope=0;
        real_t yIntersect=0;
        fitPointsSoa(&pts[0], contactIdx+1, myAddr, nSets, slope, yIntersect);
        
        fitPointsSoa(&pts[contactIdx+1], nPoints-contactIdx-1, myAddr, nSets, slope, yIntersect);
    }
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        cout << "usage: " << argv[0] << " BINARY_BLOB" << endl;
        return -1;
    }

    ifstream in(argv[1]);
    if(!in.good())
    {
        cerr << "something wrong with file" << endl;
        return -3;
    }

    my_size_t realcolumns; // number of sets
    my_size_t realrows; // number of points in each set
    in.read(reinterpret_cast<char*>(&realcolumns), sizeof(realcolumns));
    in.read(reinterpret_cast<char*>(&realrows), sizeof(realrows));
    
    vector<point_t> sets_clobbered(realcolumns*realrows);
    in.read(reinterpret_cast<char*>(sets_clobbered.data()), sizeof(point_t) * realcolumns * realrows);
    
    {
        vector<point_t> set_extended(realcolumns*10*realrows);
        for(my_size_t row=0; row<realrows; row++)
        {
            for(int k=0;k<10;k++)
            {
                memcpy(set_extended.data()+realcolumns*10*row+realcolumns*k, &ACCESS(sets_clobbered.data(), 0, realcolumns, row), sizeof(point_t)*realcolumns);
            }
        }
        std::swap(set_extended, sets_clobbered);
    }
    
    cout << "N Kaos Ksoa Caos Csoa cpy" << endl;
    for(my_size_t columns=1024; columns<=realcolumns*10; columns<<=1)
    {
        my_size_t rows=realrows;
        
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        const dim3 threads(1024);
        const dim3 grid(columns/threads.x);
        
        point_t* sets_clobbered_cuda = nullptr;
        my_size_t* cuda_contacts = nullptr;
        real_t* cuda_slopes = nullptr;
        real_t* cuda_yIntersects = nullptr;    
        if(cudaMalloc(&sets_clobbered_cuda, sizeof(*sets_clobbered_cuda) * columns*rows) != cudaSuccess) return -1;
        
        
        cudaEventRecord(start);
        cudaMemcpy(sets_clobbered_cuda, sets_clobbered.data(), sizeof(*sets_clobbered_cuda) * columns*rows, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float memcpy_time;
        cudaEventElapsedTime(&memcpy_time, start, stop);
        
        if(cudaMalloc(&cuda_contacts, sizeof(*cuda_contacts) * columns) != cudaSuccess) return -1;
        if(cudaMalloc(&cuda_slopes, sizeof(*cuda_slopes) * columns) != cudaSuccess) return -1;
        if(cudaMalloc(&cuda_yIntersects, sizeof(*cuda_yIntersects) * columns) != cudaSuccess) return -1;
    
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
        vector<real_t> slopesResults(columns);
        vector<real_t> yIntsctResults(columns);
        
        // write results back to host mem
        cudaMemcpy(contactResults.data(), cuda_contacts, sizeof(*cuda_contacts) * columns, cudaMemcpyDeviceToHost);
        cudaMemcpy(slopesResults.data(), cuda_slopes, sizeof(*cuda_slopes) * columns, cudaMemcpyDeviceToHost);
        cudaMemcpy(yIntsctResults.data(), cuda_yIntersects, sizeof(*cuda_yIntersects) * columns, cudaMemcpyDeviceToHost);
        
        duration<double, std::milli> cpuClobbered_time, cpuSoa_time;
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

        // prepare datapoints for usage on CPU
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
        }

        {
        auto t1 = high_resolution_clock::now();
        calcCpuSoa(pointsPerSet.data(), soaPoints.data(), columns);
        auto t2 = high_resolution_clock::now();
        cpuSoa_time = t2 - t1;
        }
        
        for(my_size_t i=0; i<rows; i++)
        {
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
        kernelSoa<<<grid, threads>>>(pointsPerSetCuda, soaPointsCuda, columns, cuda_contacts, cuda_slopes, cuda_yIntersects);
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
        
        
        {
        vector<my_size_t> contactResultsSoa(columns);
        vector<real_t> slopesResultsSoa(columns);
        vector<real_t> yIntsctResultsSoa(columns);
        
        // write results back to host mem
        cudaMemcpy(contactResultsSoa.data(), cuda_contacts, sizeof(*cuda_contacts) * columns, cudaMemcpyDeviceToHost);
        cudaMemcpy(slopesResultsSoa.data(), cuda_slopes, sizeof(*cuda_slopes) * columns, cudaMemcpyDeviceToHost);
        cudaMemcpy(yIntsctResultsSoa.data(), cuda_yIntersects, sizeof(*cuda_yIntersects) * columns, cudaMemcpyDeviceToHost);
        
        // assert that the results of soa kernel and clobbered kernel are same
        checkResultsSoa(contactResultsSoa.data(), slopesResultsSoa.data(), yIntsctResultsSoa.data(), contactResults.data(), slopesResults.data(), yIntsctResults.data(), columns);
        
        }
        
        cout << columns << " " << kernelClobbered_time << " " << kernelSoa_time << " " << cpuClobbered_time.count() << " " << cpuSoa_time.count() << " " << memcpy_time << endl;
    fail:
        for(my_size_t i=0; i<rows; i++)
        {
            cudaFree(soaPoints[i].z);
            cudaFree(soaPoints[i].force);
        }
        cudaFree(soaPointsCuda);
        
    fail2:
        cudaFree(pointsPerSetCuda);
        cudaFree(cuda_contacts);
        cudaFree(cuda_slopes);
        cudaFree(cuda_yIntersects);
        
    fail3:
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
    }
}
