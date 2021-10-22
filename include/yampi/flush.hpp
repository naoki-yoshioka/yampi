#ifndef YAMPI_FLUSH_HPP
# define YAMPI_FLUSH_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/window.hpp>
# include <yampi/rank.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  inline void flush(::yampi::rank const& rank, ::yampi::window const& window, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Win_flush(rank.mpi_rank(), window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::flush", environment);
  }

  inline void flush(::yampi::window const& window, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Win_flush_all(window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::flush", environment);
  }
}


#endif

