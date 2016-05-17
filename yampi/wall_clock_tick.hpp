#ifndef YAMPI_WALL_CLOCK_TICK_HPP
# define YAMPI_WALL_CLOCK_TICK_HPP

# include <mpi.h>


namespace yampi
{
  class environment;

  inline double wall_clock_tick(::yampi::environment const&)
  { return MPI_Wtick(); }
}


#endif

