
#include "gtest/gtest.h"
#include <mpi.h>
#include "MPIGrid.hpp"
#include "assert.h"
#include <iomanip>


void print_local_grid_2d(double * data, int * dims, int nrows)
{
    int rank;
    int np;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (rank != 0) 
        MPI_Recv(NULL, 0, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &status);

    std::cout << "PROCESSOR " << rank << std::endl;

    for (int i=0; i<dims[0]; i++)
    {
        if (i==nrows || i== dims[0]-nrows ) 
            for (int j=0; j<dims[1]; j++)
                std::cout << std::setw(5) << "----" << " ";
        std::cout << std::endl;

        for (int j=0; j<dims[1]; j++)
        {
            if (j==nrows || j==dims[1]-nrows) std::cout << "|";
            else std::cout << " ";
            int ind = i*dims[1] + j;
            std::cout << std::setw(4) << data[ind] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::flush;

    if (rank != np-1) 
        MPI_Send(NULL, 0, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_local_grid_3d(double * data, int * dims, int nrows)
{
    for (int i=0; i<dims[0]; i++)
    {
        std::cout << "i = " << i << std::endl;
        for (int j=0; j<dims[1]; j++)
        {
            for (int k=0; k<dims[2]; k++)
            {
                int ind = i*dims[1]*dims[2] + j*dims[2] + k;
                std::cout << std::setw(3) << data[ind] << " ";
            }
            std::cout << std::endl;
        }
    }
}

TEST(MPIGridTest, Scatter_1D)
{
    int ndims = 1;
    int global_dims[ndims];
    int local_dims[ndims];
    int np_dims[ndims];

    int rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    global_dims[0] = 100;
    np_dims[0] = np;

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, np_dims, ndims, 1, local_dims);
    ASSERT_EQ(err, 0);

    double * global_data = (double *) malloc(sizeof(double)*global_dims[0]);
    double * local_data = (double *) malloc(sizeof(double)*local_dims[0]);

    for (int i=0; i<global_dims[0]; i++)
        global_data[i] = i+1;

    grid.scatter(global_data, local_data);

    for (int i=1; i<local_dims[0]-1; i++)
        EXPECT_EQ(local_data[i], rank*global_dims[0]/np + i);
}

TEST(MPIGridTest, Share_2D)
{
    int ndims = 2;
    int global_dims[ndims];
    int np_dims[ndims];
    int local_dims[ndims];

    global_dims[0] = 4;
    global_dims[1] = 6;
    np_dims[0] = 2;
    np_dims[1] = 2;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, np_dims, ndims, 1, local_dims);
    ASSERT_EQ(err, 0);

    double * global_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);
    double * local_data = (double *) malloc(sizeof(double)*local_dims[0]*local_dims[1]);
    double * gathered_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);

    if (rank==0)
    for (int i=0; i<global_dims[0]*global_dims[1]; i++) global_data[i] = i+1;

    for (int i=0; i<local_dims[0]; i++) local_data[i] = -1;

    err = grid.scatter(global_data, local_data);
    ASSERT_EQ(err, 0);
    err = grid.share(local_data);
    ASSERT_EQ(err, 0);

    if (rank == 0) {
        EXPECT_EQ(local_data[0], 24);
        EXPECT_EQ(local_data[1], 19);
        EXPECT_EQ(local_data[2], 20);
        EXPECT_EQ(local_data[3], 21);
        EXPECT_EQ(local_data[4], 22);
        EXPECT_EQ(local_data[5], 6);
        EXPECT_EQ(local_data[10], 12);
    } else if (rank == 3) {
        EXPECT_EQ(local_data[0], 9);
        EXPECT_EQ(local_data[1], 10);
        EXPECT_EQ(local_data[2], 11);
        EXPECT_EQ(local_data[3], 12);
        EXPECT_EQ(local_data[4], 7);
        EXPECT_EQ(local_data[5], 15);
        EXPECT_EQ(local_data[10], 21);
        EXPECT_EQ(local_data[15], 3);
        EXPECT_EQ(local_data[16], 4);
    }

    err = grid.gather(gathered_data, local_data);
    ASSERT_EQ(err, 0);

    if (rank==0) {
    for (int i=0; i<global_dims[0]*global_dims[1]; i++)
        EXPECT_EQ(gathered_data[i], global_data[i]);
    }

    free(global_data);
    free(local_data);
    free(gathered_data);
}

TEST(MPIGridTest, Scatter_2D)
{
    int ndims = 2;
    int global_dims[ndims];
    int local_dims[ndims];
    int np_dims[ndims];

    int rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    global_dims[0] = 8;
    global_dims[1] = 20;
    np_dims[0] = 2;
    np_dims[1] = 2;

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, np_dims, ndims, 1, local_dims);
    ASSERT_EQ(err, 0);

    double * global_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);
    double * local_data = (double *) malloc(sizeof(double)*local_dims[0]*local_dims[1]);
    double * gathered_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);

    if (rank==0)
    for (int i=0; i<global_dims[0]*global_dims[1]; i++) global_data[i] = i+1;

    for (int i=0; i<local_dims[0]*local_dims[1]; i++) local_data[i] = -1;

    err = grid.scatter(global_data, local_data);
    ASSERT_EQ(err, 0);

    err = grid.gather(gathered_data, local_data);
    ASSERT_EQ(err, 0);

    if (rank == 0) {
        for (int i=0; i<global_dims[0]*global_dims[1]; i++)
            EXPECT_EQ(global_data[i], gathered_data[i]);
    }

    free(global_data);
    free(gathered_data);
    free(local_data);

}

TEST(MPIGridTest, TwoGhostRows)
{
    int ndims = 2;
    int global_dims[ndims];
    int local_dims[ndims];
    int np_dims[ndims];

    int rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    global_dims[0] = 10;
    global_dims[1] = 10;
    np_dims[0] = 2;
    np_dims[1] = 2;

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, np_dims, ndims, 2, local_dims);
    ASSERT_EQ(err, 0);

    double * global_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);
    double * local_data = (double *) malloc(sizeof(double)*local_dims[0]*local_dims[1]);
    double * gathered_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);

    if (rank==0)
    for (int i=0; i<global_dims[0]*global_dims[1]; i++) global_data[i] = i+1;

    for (int i=0; i<local_dims[0]*local_dims[1]; i++) local_data[i] = -1;

    err = grid.scatter(global_data, local_data);
    ASSERT_EQ(err, 0);
    grid.share(local_data);

    err = grid.gather(gathered_data, local_data);
    ASSERT_EQ(err, 0);

    if (rank == 0) {
        for (int i=0; i<global_dims[0]*global_dims[1]; i++)
            EXPECT_EQ(global_data[i], gathered_data[i]);
    }


    free(global_data);
    free(local_data);
    free(gathered_data);
}

TEST(MPIGridTest, 3D)
{
    int ndims = 3;
    int global_dims[ndims];
    int local_dims[ndims];
    int np_dims[ndims];

    int rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    global_dims[0] = 10;
    global_dims[1] = 10;
    global_dims[2] = 10;
    np_dims[0] = 2;
    np_dims[1] = 2;
    np_dims[2] = 1;

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, np_dims, ndims, 2, local_dims);
    ASSERT_EQ(err, 0);

    double * global_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]*global_dims[2]);
    double * local_data = (double *) malloc(sizeof(double)*local_dims[0]*local_dims[1]*local_dims[2]);
    double * gathered_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]*global_dims[2]);

    if (rank==0)
    for (int i=0; i<global_dims[0]*global_dims[1]*global_dims[2]; i++) global_data[i] = i+1;

    for (int i=0; i<local_dims[0]*local_dims[1]*local_dims[2]; i++) local_data[i] = -1;

    err = grid.scatter(global_data, local_data);
    ASSERT_EQ(err, 0);

    grid.share(local_data);
    //if (rank==0) print_local_grid_3d(local_data, local_dims, 2);

    err = grid.gather(gathered_data, local_data);
    ASSERT_EQ(err, 0);


    if (rank == 0) {
        for (int i=0; i<global_dims[0]*global_dims[1]*global_dims[2]; i++)
            EXPECT_EQ(global_data[i], gathered_data[i]);
    }


    free(global_data);
    free(local_data);
    free(gathered_data);
}


TEST(MPIGridTest, MultipleFields)
{
    int ndims = 2;
    int global_dims[ndims];
    int local_dims[ndims];
    int np_dims[ndims];

    int rank;
    int np;
    int nphases = 2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    global_dims[0] = 12;
    global_dims[1] = 10;
    np_dims[0] = 2;
    np_dims[1] = 2;

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, np_dims, ndims, 2, local_dims);
    ASSERT_EQ(err, 0);

    double * global_data = (double *) malloc(sizeof(double)*nphases*global_dims[0]*global_dims[1]);
    double * local_data = (double *) malloc(sizeof(double)*nphases*local_dims[0]*local_dims[1]);
    double * gathered_data = (double *) malloc(sizeof(double)*nphases*global_dims[0]*global_dims[1]);

    if (rank==0)
        for (int i=0; i<global_dims[0]*global_dims[1]*nphases; i++) global_data[i] = i+1;

    for (int i=0; i<local_dims[0]*local_dims[1]*nphases; i++) local_data[i] = -1;

    for (int i=0; i<nphases; i++)
        grid.scatter(global_data + i*global_dims[0]*global_dims[1], local_data + i*local_dims[0]*local_dims[1]);

    grid.share(local_data, nphases);

   // print_local_grid_2d(local_data, local_dims, 2);
    //print_local_grid_2d(local_data+local_dims[0]*local_dims[1], local_dims, 2);

    for (int i=0; i<nphases; i++)
        grid.gather(gathered_data + i*global_dims[0]*global_dims[1], local_data + i*local_dims[0]*local_dims[1]);

    if (rank==0) {
        for (int i=0; i<nphases*global_dims[0]*global_dims[1]; i++)
            EXPECT_EQ(global_data[i], gathered_data[i]);
    }

    free(global_data);
    free(local_data);
    free(gathered_data);
}

int main(int argc, char ** argv)
{

    ::testing::InitGoogleTest(&argc, argv);
//    ::testing::GTEST_FLAG(filter) = "MPIGridTest.MultipleFields";
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;


}
