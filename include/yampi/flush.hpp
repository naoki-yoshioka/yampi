#ifndef YAMPI_FLUSH_HPP
# define YAMPI_FLUSH_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/rank.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


# if MPI_VERSION >= 3
namespace yampi
{
  template <typename Derived>
  inline void flush(::yampi::rank const& rank, ::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Win_flush(rank.mpi_rank(), window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::flush", environment);
  }

  template <typename Derived>
  inline void flush(::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Win_flush_all(window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::flush", environment);
  }
}
# endif // MPI_VERSION >= 3


#endif

