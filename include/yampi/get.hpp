#ifndef YAMPI_GET_HPP
# define YAMPI_GET_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/buffer.hpp>
# include <yampi/target_buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  template <typename OriginValue, typename TargetValue, typename Window>
  inline void get(
    ::yampi::buffer<OriginValue> origin_buffer,
    ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const target_buffer,
    ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Get(
          origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
          window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::get", environment);
  }
}


# undef YAMPI_addressof

#endif

