
#include <string>

#include "mpi.h"

#define MPIGrid_type double
#define MPIGRID_NROWS 1

class MPIGrid {

    private:

        int m_np;
        int m_rank;
        int m_ndims;
        int * m_local_dims;
        int * m_global_dims;
        int * m_np_dims;

        MPI_Comm topology;

        void pack(double * data, double * pack, int count, int block_length, int stride);
        void unpack(double * data, double * pack, int count, int block_length, int stride);

    public:

        int setup(MPI_Comm comm, int * global_dims, int * local_dims, int * np_dims, int ndims);
        int gather(double * global_data, double * local_data);
        int scatter(double * global_data, double * local_data);
        int share(double * local_data);

};

void MPIGrid :: pack(double * data, double * packed_data, int count, int block_length, int stride)
{
    size_t num = block_length * sizeof(MPIGrid_type);

    for (int i=0; i<count; i++)
    {
        void * source = (void *) (data + i*stride);
        void * destination = (void *) (packed_data + i*block_length);
        memcpy(destination, source, num);
    }
}

void MPIGrid :: unpack(double * data, double * packed_data, int count, int block_length, int stride)
{
    size_t num = block_length * sizeof(double);

    for (int i=0; i<count; i++)
    {
        void * source = (void *) (packed_data + i*block_length);
        void * destination = (void *) (data + i*stride);
        memcpy(destination, source, num);
    }

}

int MPIGrid :: setup(MPI_Comm comm_old, int * global_dims, int * local_dims, int * np_dims, int ndims)
{

    int periodic[ndims];
    int np_product;
    MPI_Comm_size(comm_old, &m_np);

    /**
    @param comm_old the MPI communicator (usually MPI_COMM_WORLD)
    @param global_dims extents of the global system
    @param local_dims extents of the local system
    @param np_dims the number of processors in each dimensions (the decomposition)
    @param ndims the number of dimensions
    */

    /** \error ERROR 1 the number of dimensions must be greater than 0 */
    if (ndims < 1) return 1;


    /** \error ERROR 2 the number of processors in each dimensions must be greater than 0 */
    for (int i=0; i<ndims; i++) 
        if (np_dims[i] < 1) return 2;

    /** \error ERROR 3 the global dimensions must be >= MPIGRID_NROWS */
    for (int i=0; i<ndims; i++) 
        if (global_dims[i] < MPIGRID_NROWS) return 3;


    /** \error ERROR 4 the number of processors must divide evenly in global dims in each dimension */
    for (int i=0; i<ndims; i++) 
        if (global_dims[i] % np_dims[i] != 0) return 4;

    /** \error ERROR 5 the number of processors in each dimension must equal the total number of processors */
    np_product = 1;
    for (int i=0; i<ndims; i++) 
        np_product *= np_dims[i];

    if (np_product != m_np) return 5;

    // create a cartesian topology
    for (int i=0; i<ndims; i++) periodic[i] = 1;
    MPI_Cart_create(comm_old, ndims, np_dims, periodic, 0, &topology);

    MPI_Comm_rank(topology, &m_rank);

    m_ndims = ndims;
    m_global_dims = (int *) malloc(sizeof(int)*m_ndims);
    m_local_dims = (int *) malloc(sizeof(int)*m_ndims);
    m_np_dims = (int *) malloc(sizeof(int)*m_ndims);

    for (int i=0; i<ndims; i++)
    {
        local_dims[i] = global_dims[i] / np_dims[i] + MPIGRID_NROWS*2;

        m_global_dims[i] = global_dims[i];
        m_local_dims[i] = local_dims[i];
        m_np_dims[i] = np_dims[i];
    }

    return 0;
}

int MPIGrid :: scatter(double * global_data, double * local_data)
{

    MPI_Request request;
    MPI_Status status;

    int source = 0;
    int tag = 1;
    int block_length;
    int stride;
    int count;
    int offset;
    
    int subdomain[m_ndims];
    int subdomain_volume;
    int coord_stride[m_ndims];

    // calculate extents of subdomains
    subdomain_volume = 1;
    for (int i=0; i<m_ndims; i++)
    {
        subdomain[i] = m_global_dims[i] / m_np_dims[i];
        subdomain_volume *= subdomain[i];
    }

    // calculate number of contiguous chunks
    count = 1;
    for (int i=0; i<m_ndims-1; i++) count *= subdomain[i];

    double * packed_data = (double *) malloc(sizeof(double)*subdomain_volume);

    /* ============== master sends data ============= */
    if (m_rank == 0) {

        for (int i=0; i<m_ndims; i++)
        {
            coord_stride[i] = subdomain[i];
            for (int j=i+1; j<m_ndims; j++)
                coord_stride[i] *= m_np_dims[j]*subdomain[j];
        }

        block_length = subdomain[m_ndims-1];
        stride = m_global_dims[m_ndims-1];

        for (int id=0; id<m_np; id++) {

            int coords[m_ndims];
            MPI_Cart_coords(topology, id, m_ndims, coords);

            // calculate subdomain offset
            offset = 0;
            for (int i=0; i<m_ndims; i++) offset += coords[i] * coord_stride[i];

            pack(global_data+offset, packed_data, count, block_length, stride);
            MPI_Isend(packed_data, subdomain_volume, MPI_DOUBLE, id, tag, topology, &request);
        }
    }

    /* ============== everyone receives data ============= */

    offset = 0;
    for (int i=0; i<m_ndims; i++)
    {
        int local_stride_i = 1;
        for (int j=i+1; j<m_ndims; j++)
            local_stride_i *= m_local_dims[j];
        offset += local_stride_i*MPIGRID_NROWS;
    }

    block_length = subdomain[m_ndims-1];
    stride = m_local_dims[m_ndims-1];

    MPI_Recv(packed_data, subdomain_volume, MPI_DOUBLE, source, tag, topology, &status);
    unpack(local_data+offset, packed_data, count, block_length, stride);

    free(packed_data);

    return 0;
}


