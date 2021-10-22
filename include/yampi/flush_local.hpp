#ifndef YAMPI_FLUSH_LOCAL_HPP
# define YAMPI_FLUSH_LOCAL_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/window.hpp>
# include <yampi/rank.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  inline void flush_local(::yampi::rank const& rank, ::yampi::window const& window, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Win_flush_local(rank.mpi_rank(), window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::flush_local", environment);
  }

  inline void flush_local(::yampi::window const& window, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Win_flush_local_all(window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::flush_local", environment);
  }
}


#endif

