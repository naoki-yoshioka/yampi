#ifndef YAMPI_PUT_HPP
# define YAMPI_PUT_HPP

# include <type_traits>
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
  inline void put(
    ::yampi::buffer<OriginValue> const origin_buffer,
    ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
    ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Put_c(
          origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
          window.mpi_win());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Put(
          origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
          window.mpi_win());
# else // MPI_VERSION >= 3
    using value_type = typename std::remove_cv<OriginValue>::type;
    auto const error_code
      = MPI_Put(
          const_cast<value_type*>(origin_buffer.data()), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
          window.mpi_win());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::put", environment};
  }

  // Request-based put
  template <typename OriginValue, typename TargetValue, typename Window>
  inline void put(
    ::yampi::immediate_request& request,
    ::yampi::buffer<OriginValue> const origin_buffer,
    ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
    ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Rput_c(
          origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
          window.mpi_win(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Rput(
          origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
          target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
          window.mpi_win(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::put", environment};
    request.reset(mpi_request, environment);
  }
}


#endif

