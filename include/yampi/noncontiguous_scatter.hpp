#ifndef YAMPI_NONCONTIGUOUS_SCATTER_HPP
# define YAMPI_NONCONTIGUOUS_SCATTER_HPP

# include <memory>

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/noncontiguous_buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/in_place.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>
# include <yampi/root_call_on_nonroot_error.hpp>


namespace yampi
{
  // only for intracommunicators
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 3
    auto const error_code
      = MPI_Scatterv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Scatterv(
          const_cast<SendValue*>(send_buffer.data()), const_cast<int*>(send_buffer.count_first()),
          const_cast<int*>(send_buffer.displacement_first()), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::noncontiguous_scatter", environment);
  }

  template <typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error("yampi::noncontiguous_scatter");

    auto const error_code
      = MPI_Scatterv(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::noncontiguous_scatter", environment);
  }

  template <typename Value>
  inline void noncontiguous_scatter(
    ::yampi::in_place_t const,
    ::yampi::noncontiguous_buffer<Value, false> const send_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) != root)
      throw ::yampi::root_call_on_nonroot_error("yampi::noncontiguous_scatter");

# if MPI_VERSION >= 3
    auto const error_code
      = MPI_Scatterv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Scatterv(
          const_cast<SendValue*>(send_buffer.data()), const_cast<int*>(send_buffer.count_first()),
          const_cast<int*>(send_buffer.displacement_first()), send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::noncontiguous_scatter", environment);
  }

  // only for intercommunicators
  template <typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    auto const error_code
      = MPI_Scatterv(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::noncontiguous_scatter", environment);
  }

  template <typename SendValue>
  inline void noncontiguous_scatter(
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 3
    auto const error_code
      = MPI_Scatterv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Scatterv(
          const_cast<SendValue*>(send_buffer.data()), const_cast<int*>(send_buffer.count_first()),
          const_cast<int*>(send_buffer.displacement_first()), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::noncontiguous_scatter", environment);
  }

  inline void noncontiguous_scatter(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    auto const error_code
      = MPI_Scatterv(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, nullptr, 0, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::noncontiguous_scatter", environment);
  }
}


#endif
