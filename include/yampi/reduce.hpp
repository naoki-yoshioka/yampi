#ifndef YAMPI_REDUCE_HPP
# define YAMPI_REDUCE_HPP

# include <cassert>
# include <type_traits>
# include <iterator>
# include <memory>

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/intercommunicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/in_place.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>


namespace yampi
{
  // only for intracommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void reduce(
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      (std::is_same<
         typename std::iterator_traits<ContiguousIterator>::value_type,
         SendValue>::value),
      "value_type of ContiguousIterator must be the same to SendValue");
# if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() <= send_buffer.data());
# else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() <= send_buffer.data());
# endif // MPI_VERSION >= 4

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Reduce_c(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Reduce(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm());
# else //MPI_VERSION >= 3
    auto const error_code
      = MPI_Reduce(
          const_cast<SendValue*>(send_buffer.data()), std::addressof(*first),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm());
# endif //MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::reduce", environment);
  }

  template <typename SendValue>
  inline void reduce(
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error("yampi::reduce");

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Reduce_c(
          send_buffer.data(), nullptr,
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Reduce(
          send_buffer.data(), nullptr,
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm());
# else //MPI_VERSION >= 3
    auto const error_code
      = MPI_Reduce(
          const_cast<SendValue*>(send_buffer.data()), nullptr,
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm());
# endif //MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::reduce", environment);
  }

  template <typename Value>
  inline void reduce(
    ::yampi::in_place_t const, ::yampi::buffer<Value> buffer,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = communicator.rank(environment) == root
        ? MPI_Reduce_c(
            MPI_IN_PLACE, buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm())
        : MPI_Reduce_c(
            buffer.data(), nullptr, buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = communicator.rank(environment) == root
        ? MPI_Reduce(
            MPI_IN_PLACE, buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm())
        : MPI_Reduce(
            buffer.data(), nullptr, buffer.count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::reduce", environment);
  }

  // only for intercommunicators
  template <typename SendValue>
  inline void reduce(
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Reduce_c(
          send_buffer.data(), nullptr,
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Reduce(
          send_buffer.data(), nullptr,
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm());
# else //MPI_VERSION >= 3
    auto const error_code
      = MPI_Reduce(
          const_cast<SendValue*>(send_buffer.data()), nullptr,
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm());
# endif //MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::reduce", environment);
  }

  template <typename ReceiveValue>
  inline void reduce(
    ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Reduce_c(
          nullptr, receive_buffer.data(),
          receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), MPI_ROOT, communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Reduce(
          nullptr, receive_buffer.data(),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), MPI_ROOT, communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::reduce", environment);
  }

  inline void reduce(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Reduce_c(
          nullptr, nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          MPI_OP_NULL, MPI_PROC_NULL, communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Reduce(
          nullptr, nullptr, 0, MPI_DATATYPE_NULL,
          MPI_OP_NULL, MPI_PROC_NULL, communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::reduce", environment);
  }
}


#endif
