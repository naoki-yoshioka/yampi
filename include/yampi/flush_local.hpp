#ifndef YAMPI_FLUSH_LOCAL_HPP
# define YAMPI_FLUSH_LOCAL_HPP

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
  inline void flush_local(::yampi::rank const& rank, ::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Win_flush_local(rank.mpi_rank(), window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::flush_local", environment);
  }

  template <typename Derived>
  inline void flush_local(::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Win_flush_local_all(window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::flush_local", environment);
  }
}
# endif // MPI_VERSION >= 3


#endif

