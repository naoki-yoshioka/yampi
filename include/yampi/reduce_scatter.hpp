#ifndef YAMPI_REDUCE_SCATTER_HPP
# define YAMPI_REDUCE_SCATTER_HPP

# include <cassert>
# include <type_traits>
# include <iterator>
# include <memory>

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/in_place.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  template <typename SendValue, typename ContiguousIterator>
  inline void reduce_scatter(
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      (std::is_same<
         typename std::iterator_traits<ContiguousIterator>::value_type,
         SendValue>::value),
      "value_type of ContiguousIterator must be the same to Value");
    assert(send_buffer.data() != std::addressof(*first));
    assert(send_buffer.count() == communicator.size(environment));

# if MPI_VERSION >= 3
    int const error_code
      = MPI_Reduce_scatter_block(
          send_buffer.data(), std::addressof(*first), 1, send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Reduce_scatter_block(
          const_cast<SendValue*>(send_buffer.data()), std::addressof(*first), 1, send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::reduce_scatter", environment);
  }

  template <typename Value>
  inline void reduce_scatter(
    ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    assert(send_buffer.data() != receive_buffer.data());
    assert(send_buffer.count() == communicator.size(environment) * receive_buffer.count());
    assert(send_buffer.datatype() == receive_buffer.datatype());

# if MPI_VERSION >= 3
    int const error_code
      = MPI_Reduce_scatter_block(
          send_buffer.data(), receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Reduce_scatter_block(
          const_cast<SendValue*>(send_buffer.data()), receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::reduce_scatter", environment);
  }

  template <typename Value>
  inline void reduce_scatter(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value> receive_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 3
    int const error_code
      = MPI_Reduce_scatter_block(
          MPI_IN_PLACE, receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Reduce_scatter_block(
          MPI_IN_PLACE, receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::reduce_scatter", environment);
  }
}


#endif
