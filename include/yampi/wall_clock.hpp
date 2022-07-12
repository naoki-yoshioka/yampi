#ifndef YAMPI_WALL_CLOCK_HPP
# define YAMPI_WALL_CLOCK_HPP

# include <chrono>
# include <ratio>

# include <mpi.h>

# include <yampi/environment.hpp>


namespace yampi
{
  struct wall_clock
  {
    typedef double rep;
    typedef std::ratio<1> period;
    typedef std::chrono::duration<rep, period> duration;
    typedef std::chrono::time_point< ::yampi::wall_clock > time_point;
    static constexpr bool is_steady = false;

    static time_point now(::yampi::environment const&)
    { return static_cast<time_point>(static_cast<duration>(MPI_Wtime())); }
    static duration tick(::yampi::environment const&)
    { return static_cast<duration>(MPI_Wtick()); }
  };
}


#endif

