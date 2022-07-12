#ifndef YAMPI_ACCUMULATE_HPP
# define YAMPI_ACCUMULATE_HPP

# include <memory>

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/buffer.hpp>
# include <yampi/target_buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  template <typename OriginValue, typename TargetValue, typename Window>
  inline void accumulate(
    ::yampi::buffer<OriginValue> const origin_buffer,
    ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Accumulate(
          origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), window.mpi_win());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::accumulate", environment);
  }
}


#endif

