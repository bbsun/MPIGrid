
#include <mpi.h>
#include "MPIGrid.hpp"

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int global_dims[3];
    int local_dims[3];
    int np_dims[3];

    global_dims[0] = 2;
    global_dims[1] = 3;
    global_dims[2] = 4;

    np_dims[0] = 2;
    np_dims[1] = 1;
    np_dims[2] = 2;


    MPIGrid grid;
    grid.setup(MPI_COMM_WORLD, global_dims, local_dims, np_dims, 3);

    MPI_Finalize();

    return 0;
}
