#ifndef YAMPI_ALL_REDUCE_HPP
# define YAMPI_ALL_REDUCE_HPP

# include <cassert>
# include <type_traits>
# include <iterator>
# include <memory>

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/rank.hpp>
# include <yampi/in_place.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  template <typename SendValue, typename ContiguousIterator>
  inline void all_reduce(
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
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
      = MPI_Allreduce_c(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Allreduce(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Allreduce(
          const_cast<SendValue*>(send_buffer.data()), std::addressof(*first),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::all_reduce", environment);
  }

  // only for blocking all_reduce
  template <typename SendValue>
  inline SendValue all_reduce(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::binary_operation const& operation,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    assert(send_buffer.count().mpi_count() == 1);
# else // MPI_VERSION >= 4
    assert(send_buffer.count() == 1);
# endif // MPI_VERSION >= 4

    SendValue result;
    ::yampi::all_reduce(send_buffer, &result, operation, communicator, environment);
    return result;
  }

  template <typename Value>
  inline void all_reduce(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value> buffer, ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Allreduce_c(
          MPI_IN_PLACE,
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Allreduce(
          MPI_IN_PLACE,
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::all_reduce", environment);
  }
}


#endif
