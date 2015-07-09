
#include <string>

#include "mpi.h"

#define MPIGrid_type double
#define MPIGRID_NROWS 1

class MPIGrid {

    private:

        MPI_Comm topology;

        void pack(double * data, double * pack, int count, int block_length, int stride);

    public:

        int setup(MPI_Comm comm, int * global_dims, int * local_dims, int * np_dims, int ndims);
        int gather(double * global_data, double * local_data);
        int scatter(double * global_data, double * local_data);
        int share(double * local_data);

};

void MPIGrid :: pack(double * data, double * pack, int count, int block_length, int stride)
{
    size_t num = block_length * sizeof(MPIGrid_type);

    for (int i=0; i<count; i++)
    {
        void * source = (void *) &(data[stride*i]);
        void * destination = (void *) &(pack[i*block_length]);
        memcpy(destination, source, num);
    }
}

int MPIGrid :: setup(MPI_Comm comm_old, int * global_dims, int * local_dims, int * np_dims, int ndims)
{

    int periodic[ndims];
    int np, rank;
    int np_product;

    MPI_Comm_rank(comm_old, &rank);
    MPI_Comm_size(comm_old, &np);


    // check that number of processors fits evenly in each dimension
    for (int i=0; i<ndims; i++) 
        if (global_dims[i] % np_dims[i] != 0) 
            return 1;

    // check that the processors in each dimension multiply to the correct total
    np_product = 1;
    for (int i=0; i<ndims; i++) 
        np_product *= np_dims[i];

    if (np_product != np) return 2;

    // create a cartesian topology
    for (int i=0; i<ndims; i++) periodic[i] = 1;
    MPI_Cart_create(comm_old, ndims, np_dims, periodic, 1, &topology);

    for (int i=0; i<ndims; i++)
        local_dims[i] = global_dims[i] / np_dims[i] + MPIGRID_NROWS*2;

    return 0;
}

int MPIGrid :: scatter(double * global_data, double * local_data)
{
    return 0;
}


