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
    assert(send_buffer.data() != std::addressof(*first));

# if MPI_VERSION >= 3
    int const error_code
      = MPI_Allreduce(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
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
    assert(send_buffer.count() == 1);

    SendValue result;
    ::yampi::all_reduce(send_buffer, &result, operation, communicator, environment);
    return result;
  }

  template <typename Value>
  inline void all_reduce(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value> receive_buffer, ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Allreduce(
          MPI_IN_PLACE,
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::all_reduce", environment);
  }
}


#endif
