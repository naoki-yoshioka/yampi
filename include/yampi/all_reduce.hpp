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
# include <yampi/immediate_request.hpp>
# if MPI_VERSION >= 4
#   include <yampi/persistent_request.hpp>
#   include <yampi/information.hpp>
# endif // MPI_VERSION >= 4


namespace yampi
{
  // Blocking all-reduce
  template <typename SendValue, typename ContiguousIterator>
  inline void all_reduce(
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
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
      throw ::yampi::error{error_code, "yampi::all_reduce", environment};
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
    ::yampi::all_reduce(send_buffer, std::addressof(result), operation, communicator, environment);
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
      throw ::yampi::error{error_code, "yampi::all_reduce", environment};
  }

  // Nonblocking all-reduce
  template <typename SendValue, typename ContiguousIterator>
  inline void all_reduce(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
# if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() <= send_buffer.data());
# else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() <= send_buffer.data());
# endif // MPI_VERSION >= 4

    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Iallreduce_c(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Iallreduce(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::all_reduce", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void all_reduce(
    ::yampi::in_place_t const,
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Iallreduce_c(
          MPI_IN_PLACE,
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Iallreduce(
          MPI_IN_PLACE,
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::all_reduce", environment};
    request.reset(mpi_request, environment);
  }
# if MPI_VERSION >= 4

  // Persistent all-reduce
  template <typename SendValue, typename ContiguousIterator>
  inline void all_reduce(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::binary_operation const& operation, ::yampi::information const& information,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() <= send_buffer.data());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Allreduce_init_c(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::all_reduce", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void all_reduce(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::binary_operation const& operation,
    ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Allreduce_init_c(
          MPI_IN_PLACE,
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::all_reduce", environment};
    request.reset(mpi_request, environment);
  }

  // information omitted
  template <typename SendValue, typename ContiguousIterator>
  inline void all_reduce(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() <= send_buffer.data());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Allreduce_init_c(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::all_reduce", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void all_reduce(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Allreduce_init_c(
          MPI_IN_PLACE,
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::all_reduce", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 4
}


#endif
