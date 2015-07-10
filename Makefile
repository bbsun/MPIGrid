
gtest = /Volumes/DATA/Software/gtest-1.7.0

default:
	mpic++ -isystem $(gtest)/include -I$(gtest) -pthread -c $(gtest)/src/gtest-all.cc
	ar -rv libgtest.a gtest-all.o
	mpic++ -Wall -isystem $(gtest)/include -pthread test_MPIGrid.cpp libgtest.a -o a.out
