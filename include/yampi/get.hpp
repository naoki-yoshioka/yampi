#ifndef YAMPI_GET_HPP
# define YAMPI_GET_HPP

# include <memory>

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/buffer.hpp>
# include <yampi/target_buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/immediate_request.hpp>


namespace yampi
{
  template <typename OriginValue, typename TargetValue, typename Window>
  inline void get(
    ::yampi::buffer<OriginValue> origin_buffer,
    ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
    ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Get_c(
          origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
          window.mpi_win());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Get(
          origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
          window.mpi_win());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::get", environment};
  }

  // Request-based get
  template <typename OriginValue, typename TargetValue, typename Window>
  inline void get(
    ::yampi::immediate_request& request,
    ::yampi::buffer<OriginValue> origin_buffer,
    ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
    ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Rget_c(
          origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
          window.mpi_win(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Rget(
          origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
          window.mpi_win(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::get", environment};
    request.reset(mpi_request, environment);
  }
}


#endif

