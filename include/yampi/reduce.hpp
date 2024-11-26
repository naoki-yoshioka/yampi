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
# include <yampi/immediate_request.hpp>
# if MPI_VERSION >= 4
#   include <yampi/persistent_request.hpp>
#   include <yampi/information.hpp>
# endif // MPI_VERSION >= 4


namespace yampi
{
  // Blocking reduce
  // only for intracommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void reduce(
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
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
      throw ::yampi::error{error_code, "yampi::reduce", environment};
  }

  template <typename SendValue>
  inline void reduce(
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error{"yampi::reduce"};

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
      throw ::yampi::error{error_code, "yampi::reduce", environment};
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
      throw ::yampi::error{error_code, "yampi::reduce", environment};
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
      throw ::yampi::error{error_code, "yampi::reduce", environment};
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
      throw ::yampi::error{error_code, "yampi::reduce", environment};
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
      throw ::yampi::error{error_code, "yampi::reduce", environment};
  }

  // Nonblocking reduce
  // only for intracommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void reduce(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
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
      = MPI_Ireduce_c(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ireduce(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline void reduce(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error{"yampi::reduce"};

    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ireduce_c(
          send_buffer.data(), nullptr,
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ireduce(
          send_buffer.data(), nullptr,
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void reduce(
    ::yampi::in_place_t const,
    ::yampi::immediate_request& request, ::yampi::buffer<Value> buffer,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = communicator.rank(environment) == root
        ? MPI_Ireduce_c(
            MPI_IN_PLACE, buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
            std::addressof(mpi_request))
        : MPI_Ireduce_c(
            buffer.data(), nullptr, buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = communicator.rank(environment) == root
        ? MPI_Ireduce(
            MPI_IN_PLACE, buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
            std::addressof(mpi_request))
        : MPI_Ireduce(
            buffer.data(), nullptr, buffer.count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  // only for intercommunicators
  template <typename SendValue>
  inline void reduce(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ireduce_c(
          send_buffer.data(), nullptr,
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ireduce(
          send_buffer.data(), nullptr,
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  template <typename ReceiveValue>
  inline void reduce(
    ::yampi::immediate_request& request,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::binary_operation const& operation,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ireduce_c(
          nullptr, receive_buffer.data(),
          receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), MPI_ROOT, communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ireduce(
          nullptr, receive_buffer.data(),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), MPI_ROOT, communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  inline void reduce(
    ::yampi::immediate_request& request,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ireduce_c(
          nullptr, nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          MPI_OP_NULL, MPI_PROC_NULL, communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ireduce(
          nullptr, nullptr, 0, MPI_DATATYPE_NULL,
          MPI_OP_NULL, MPI_PROC_NULL, communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }
# if MPI_VERSION >= 4

  // Persistent reduce
  // only for intracommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void reduce(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() <= send_buffer.data());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Reduce_init_c(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(),
          std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline void reduce(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error{"yampi::reduce"};

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Reduce_init_c(
          send_buffer.data(), nullptr,
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(),
          std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void reduce(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request, ::yampi::buffer<Value> buffer,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = communicator.rank(environment) == root
        ? MPI_Reduce_init_c(
            MPI_IN_PLACE, buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(),
            std::addressof(mpi_request))
        : MPI_Reduce_init_c(
            buffer.data(), nullptr, buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(),
            std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  // only for intercommunicators
  template <typename SendValue>
  inline void reduce(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Reduce_init_c(
          send_buffer.data(), nullptr,
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(),
          std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  template <typename ReceiveValue>
  inline void reduce(
    ::yampi::persistent_request& request,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::binary_operation const& operation,
    ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Reduce_init_c(
          nullptr, receive_buffer.data(),
          receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), MPI_ROOT, communicator.mpi_comm(), information.mpi_info(),
          std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  inline void reduce(
    ::yampi::persistent_request& request, ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Reduce_init_c(
          nullptr, nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          MPI_OP_NULL, MPI_PROC_NULL, communicator.mpi_comm(), information.mpi_info(),
          std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  // information omitted
  // only for intracommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void reduce(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() <= send_buffer.data());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Reduce_init_c(
          send_buffer.data(), std::addressof(*first),
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL,
          std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline void reduce(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error{"yampi::reduce"};

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Reduce_init_c(
          send_buffer.data(), nullptr,
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL,
          std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void reduce(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request, ::yampi::buffer<Value> buffer,
    ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = communicator.rank(environment) == root
        ? MPI_Reduce_init_c(
            MPI_IN_PLACE, buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL,
            std::addressof(mpi_request))
        : MPI_Reduce_init_c(
            buffer.data(), nullptr, buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL,
            std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  // only for intercommunicators
  template <typename SendValue>
  inline void reduce(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::binary_operation const& operation, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Reduce_init_c(
          send_buffer.data(), nullptr,
          send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL,
          std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  template <typename ReceiveValue>
  inline void reduce(
    ::yampi::persistent_request& request,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::binary_operation const& operation,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Reduce_init_c(
          nullptr, receive_buffer.data(),
          receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), MPI_ROOT, communicator.mpi_comm(), MPI_INFO_NULL,
          std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }

  inline void reduce(
    ::yampi::persistent_request& request,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Reduce_init_c(
          nullptr, nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          MPI_OP_NULL, MPI_PROC_NULL, communicator.mpi_comm(), MPI_INFO_NULL,
          std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::reduce", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 4
}


#endif
