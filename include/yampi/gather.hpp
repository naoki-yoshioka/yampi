#ifndef YAMPI_GATHER_HPP
# define YAMPI_GATHER_HPP

# include <cassert>
# include <type_traits>
# include <iterator>
# include <memory>

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/intercommunicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/in_place.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>
# include <yampi/root_call_on_nonroot_error.hpp>


namespace yampi
{
  // only for intracommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void gather(
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      (std::is_same<
         typename std::iterator_traits<ContiguousIterator>::value_type,
         SendValue>::value),
      "value_type of ContiguousIterator must be the same to SendValue");
    assert(communicator.rank(environment) != root or (send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * communicator.size(environment) <= send_buffer.data()));

# if MPI_VERSION >= 3
    auto const error_code
      = MPI_Gather(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Gather(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void gather(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    assert(communicator.rank(environment) != root or (send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data()));

    auto const size = communicator.size(environment);
    auto const receive_count = receive_buffer.count() / size;
    assert(receive_count * size == receive_buffer.count());

# if MPI_VERSION >= 3
    auto const error_code
      = MPI_Gather(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Gather(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather", environment);
  }

  template <typename SendValue>
  inline void gather(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error("yampi::gather");

# if MPI_VERSION >= 3
    auto const error_code
      = MPI_Gather(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Gather(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather", environment);
  }

  template <typename Value>
  inline void gather(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) != root)
      throw ::yampi::root_call_on_nonroot_error("yampi::gather");

    auto const size = communicator.size(environment);
    auto const receive_count = receive_buffer.count() / size;
    assert(receive_count * size == receive_buffer.count());

    auto const error_code
      = MPI_Gather(
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather", environment);
  }

  // only for intercommunicators
  template <typename SendValue>
  inline void gather(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 3
    auto const error_code
      = MPI_Gather(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Gather(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather", environment);
  }

  template <typename ReceiveValue>
  inline void gather(
    ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    auto const remote_size = communicator.remote_size(environment);
    auto const receive_count = receive_buffer.count() / remote_size;
    assert(receive_count * remote_size == receive_buffer.count());

    auto const error_code
      = MPI_Gather(
          nullptr, 0, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather", environment);
  }

  template <typename ReceiveValue>
  inline void gather(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    auto const error_code
      = MPI_Gather(
          nullptr, 0, MPI_DATATYPE_NULL, nullptr, 0, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather", environment);
  }
}


#endif

