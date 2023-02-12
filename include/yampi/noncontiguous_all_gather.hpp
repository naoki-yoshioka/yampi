#ifndef YAMPI_NONCONTIGUOUS_ALL_GATHER_HPP
# define YAMPI_NONCONTIGUOUS_ALL_GATHER_HPP

# include <memory>

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/noncontiguous_buffer.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/in_place.hpp>
# if MPI_VERSION >= 3
#   include <yampi/topology.hpp>
# endif
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_all_gather(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 3
    auto const error_code
      = MPI_Allgatherv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Allgatherv(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::noncontiguous_all_gather", environment);
  }

  // only for intracommunicators
  template <typename Value>
  inline void noncontiguous_all_gather(
    ::yampi::in_place_t const,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    auto const error_code
      = MPI_Allgatherv(
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::noncontiguous_all_gather", environment);
  }
# if MPI_VERSION >= 3

  // neighbor noncontiguous_all_gather
  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_all_gather(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    auto const error_code
      = MPI_Neighbor_allgatherv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::noncontiguous_all_gather", environment);
  }
# endif // MPI_VERSION >= 3
}


#endif
