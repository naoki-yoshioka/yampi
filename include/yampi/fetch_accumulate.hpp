#ifndef YAMPI_FETCH_ACCUMULATE_HPP
# define YAMPI_FETCH_ACCUMULATE_HPP

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
# include <yampi/binary_operation.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


# if MPI_VERSION >= 3
namespace yampi
{
  template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
  inline void fetch_accumulate(
    ::yampi::buffer<OriginValue> const origin_buffer,
    ::yampi::buffer<ResultValue> result_buffer,
    ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Get_accumulate(
          origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          result_buffer.data(), result_buffer.count(), result_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::fetch_accumulate", environment);
  }
}
# endif // MPI_VERSION >= 3


# undef YAMPI_addressof

#endif

