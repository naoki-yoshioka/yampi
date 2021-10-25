#ifndef YAMPI_PUT_HPP
# define YAMPI_PUT_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/target_buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/request.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  template <typename OriginValue, typename TargetValue, typename Derived>
  inline void put(
    ::yampi::buffer<OriginValue> const& origin_buffer,
    ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
    ::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Put(
          origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
          window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::put", environment);
  }
# if MPI_VERSION >= 3

  // Request-based put
  template <typename OriginValue, typename TargetValue, typename Derived>
  inline void put(
    ::yampi::request& request,
    ::yampi::buffer<OriginValue> const& origin_buffer,
    ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
    ::yampi::window_base<Derived> const& window, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    int const error_code
      = MPI_Rput(
          origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
          window.mpi_win(), YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::put", environment);

    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 3
}


# undef YAMPI_addressof

#endif

