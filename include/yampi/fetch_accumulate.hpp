#ifndef YAMPI_FETCH_ACCUMULATE_HPP
# define YAMPI_FETCH_ACCUMULATE_HPP

# include <cassert>
# include <memory>

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/buffer.hpp>
# include <yampi/target_buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/immediate_request.hpp>


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
    assert(origin_buffer.data() != result_buffer.data());

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Get_accumulate_c(
          origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
          result_buffer.data(), result_buffer.count().mpi_count(), result_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), window.mpi_win());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Get_accumulate(
          origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          result_buffer.data(), result_buffer.count(), result_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), window.mpi_win());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::fetch_accumulate", environment};
  }

  // Request-based fetch-accumulate
  template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
  inline void fetch_accumulate(
    ::yampi::immediate_request& request,
    ::yampi::buffer<OriginValue> const origin_buffer,
    ::yampi::buffer<ResultValue> result_buffer,
    ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
  {
    assert(origin_buffer.data() != result_buffer.data());

    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Rget_accumulate_c(
          origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
          result_buffer.data(), result_buffer.count().mpi_count(), result_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), window.mpi_win(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Rget_accumulate(
          origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          result_buffer.data(), result_buffer.count(), result_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), window.mpi_win(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::fetch_accumulate", environment};
    request.reset(mpi_request, environment);
  }
}
# endif // MPI_VERSION >= 3


#endif

