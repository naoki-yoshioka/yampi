#ifndef YAMPI_WALL_CLOCK_TIME_HPP
# define YAMPI_WALL_CLOCK_TIME_HPP

# include <mpi.h>


namespace yampi
{
  class environment;

  inline double wall_clock_time(::yampi::environment const&)
  { return MPI_Wtime(); }
}


#endif

