#ifndef YAMPI_IMMEDIATE_REQUEST_HPP
# define YAMPI_IMMEDIATE_REQUEST_HPP

# include <cassert>
# include <utility>
# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/noncontiguous_buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/communication_mode.hpp>
# include <yampi/message.hpp>
# include <yampi/in_place.hpp>
# include <yampi/request_base.hpp>
# if MPI_VERSION >= 3
#   include <yampi/communicator.hpp>
#   include <yampi/intercommunicator.hpp>
#   include <yampi/topology.hpp>
# endif
# if MPI_VERSION >= 4
#   include <yampi/information.hpp>
# endif
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>
# include <yampi/root_call_on_nonroot_error.hpp>


namespace yampi
{
  namespace immediate_request_detail
  {
    // send
    template <typename Value>
    inline void standard_send(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      auto const error_code
        = MPI_Isend_c(
            buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# elif MPI_VERSION >= 3
      auto const error_code
        = MPI_Isend(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# else // MPI_VERSION
      auto const error_code
        = MPI_Isend(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# endif // MPI_VERSION
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::standard_send", environment);
    }

    template <typename Value>
    inline void buffered_send(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ibsend_c(
            buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# elif MPI_VERSION >= 3
      auto const error_code
        = MPI_Ibsend(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# else // MPI_VERSION
      auto const error_code
        = MPI_Ibsend(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# endif // MPI_VERSION
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::buffered_send", environment);
    }

    template <typename Value>
    inline void synchronous_send(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      auto const error_code
        = MPI_Issend_c(
            buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# elif MPI_VERSION >= 3
      auto const error_code
        = MPI_Issend(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# else // MPI_VERSION
      auto const error_code
        = MPI_Issend(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# endif // MPI_VERSION
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::synchronous_send", environment);
    }

    template <typename Value>
    inline void ready_send(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      auto const error_code
        = MPI_Irsend_c(
            buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# elif MPI_VERSION >= 3
      auto const error_code
        = MPI_Irsend(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# else // MPI_VERSION
      auto const error_code
        = MPI_Irsend(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# endif // MPI_VERSION
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::ready_send", environment);
    }

    // receive
    template <typename Value>
    inline void receive(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      auto const error_code
        = MPI_Irecv_c(
            buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Irecv(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::receive", environment);
    }
# if MPI_VERSION >= 3

    template <typename Value>
    inline void receive(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> buffer, ::yampi::message& message,
      ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Imrecv_c(
            buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            std::addressof(message.mpi_message()), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Imrecv(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            std::addressof(message.mpi_message()), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::receive", environment);
    }

#   if MPI_VERSION >= 4
    // send_receive
    template <typename SendValue, typename ReceiveValue>
    inline void send_receive(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Isendrecv_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), send_tag.mpi_tag(),
            receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
            source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), std::addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::send_receive", environment);
    }

    // with replacement
    template <typename Value>
    inline void send_receive(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Isendrecv_replace_c(
            buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), send_tag.mpi_tag(),
            source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), std::addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::send_receive", environment);
    }
#   endif // MPI_VERSION >= 4

    // Nonblocking collective operations
    // barrier
    inline void barrier(
      MPI_Request& mpi_request,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      auto const error_code
        = MPI_Ibarrier(communicator.mpi_comm(), std::addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::barrier", environment);
    }

    // broadcast
    template <typename Value>
    inline void broadcast(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> buffer, ::yampi::rank const root,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ibcast_c(
            buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ibcast(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::broadcast", environment);
    }

    template <typename SendValue>
    inline void broadcast(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ibcast_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ibcast(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::broadcast", environment);
    }

    inline void broadcast(
      MPI_Request& mpi_request,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ibcast_c(
            nullptr, MPI_Count{0}, MPI_DATATYPE_NULL, MPI_PROC_NULL, communicator.mpi_comm(),
            std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ibcast(
            nullptr, 0, MPI_DATATYPE_NULL, MPI_PROC_NULL, communicator.mpi_comm(),
            std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::broadcast", environment);
    }

    // gather
    template <typename SendValue, typename ContiguousIterator>
    inline void gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");
#   if MPI_VERSION >= 4
      assert(communicator.rank(environment) != root or (send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.size(environment) <= send_buffer.data()));
#   else // MPI_VERSION >= 4
      assert(communicator.rank(environment) != root or (send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * communicator.size(environment) <= send_buffer.data()));
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::gather", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    inline void gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      assert(communicator.rank(environment) != root or (send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data()));
#   else // MPI_VERSION >= 4
      assert(communicator.rank(environment) != root or (send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data()));
#   endif // MPI_VERSION >= 4

      auto const size = communicator.size(environment);
      auto const receive_count = receive_buffer.count() / size;
      assert(receive_count * size == receive_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::gather", environment);
    }

    template <typename SendValue>
    inline void gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) == root)
        throw ::yampi::nonroot_call_on_root_error("yampi::immediate_request_detail::gather");

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            nullptr, 0, MPI_DATATYPE_NULL,
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::gather", environment);
    }

    template <typename Value>
    inline void gather_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) != root)
        throw ::yampi::root_call_on_nonroot_error("yampi::immediate_request_detail::gather_in_place");

      auto const size = communicator.size(environment);
      auto const receive_count = receive_buffer.count() / size;
      assert(receive_count * size == receive_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather_c(
            MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::gather_in_place", environment);
    }

    template <typename SendValue>
    inline void gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            nullptr, 0, MPI_DATATYPE_NULL,
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::gather", environment);
    }

    template <typename ReceiveValue>
    inline void gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      auto const remote_size = communicator.remote_size(environment);
      auto const receive_count = receive_buffer.count() / remote_size;
      assert(receive_count * remote_size == receive_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather_c(
            nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
            MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather(
            nullptr, 0, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
            MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::gather", environment);
    }

    inline void gather(
      MPI_Request& mpi_request,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather_c(
            nullptr, MPI_Count{0}, MPI_DATATYPE_NULL, nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
            MPI_PROC_NULL, communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igather(
            nullptr, 0, MPI_DATATYPE_NULL, nullptr, 0, MPI_DATATYPE_NULL,
            MPI_PROC_NULL, communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::gather", environment);
    }

    // noncontiguous_gather
    template <typename SendValue, typename ReceiveValue>
    inline void noncontiguous_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igatherv_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(),
            reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
            receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igatherv(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count_first(),
            receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_gather", environment);
    }

    template <typename SendValue>
    inline void noncontiguous_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) == root)
        throw ::yampi::nonroot_call_on_root_error("yampi::immediate_request_detail::noncontiguous_gather");

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igatherv_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igatherv(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_gather", environment);
    }

    template <typename Value>
    inline void noncontiguous_gather_in_place(
      MPI_Request& mpi_request,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) != root)
        throw ::yampi::root_call_on_nonroot_error("yampi::immediate_request_detail::noncontiguous_gather_in_place");

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igatherv_c(
            MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
            receive_buffer.data(),
            reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
            receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igatherv(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count_first(),
            receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_gather_in_place", environment);
    }

    template <typename SendValue>
    inline void noncontiguous_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igatherv_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igatherv(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_gather", environment);
    }

    template <typename ReceiveValue>
    inline void noncontiguous_gather(
      MPI_Request& mpi_request,
      ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igatherv_c(
            nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
            receive_buffer.data(),
            reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
            receive_buffer.datatype().mpi_datatype(),
            MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igatherv(
            nullptr, 0, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count_first(),
            receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
            MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_gather", environment);
    }

    inline void noncontiguous_gather(
      MPI_Request& mpi_request,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Igatherv_c(
            nullptr, MPI_Count{0}, MPI_DATATYPE_NULL, nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            MPI_PROC_NULL, communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Igatherv(
            nullptr, 0, MPI_DATATYPE_NULL, nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            MPI_PROC_NULL, communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_gather", environment);
    }

    // scatter
    template <typename ContiguousIterator, typename ReceiveValue>
    inline void scatter(
      MPI_Request& mpi_request,
      ContiguousIterator const first, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           ReceiveValue>::value),
        "value_type of ContiguousIterator must be the same to ReceiveValue");
#   if MPI_VERSION >= 4
      assert(communicator.rank(environment) != root or (std::addressof(*first) + receive_buffer.count().mpi_count() * communicator.size(environment) <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= std::addressof(*first)));
#   else // MPI_VERSION >= 4
      assert(communicator.rank(environment) != root or (std::addressof(*first) + receive_buffer.count() * communicator.size(environment) <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= std::addressof(*first)));
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter_c(
            std::addressof(*first), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter(
            std::addressof(*first), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::scatter", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    inline void scatter(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      assert(communicator.rank(environment) != root or (send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data()));
#   else // MPI_VERSION >= 4
      assert(communicator.rank(environment) != root or (send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data()));
#   endif // MPI_VERSION >= 4

      auto const size = communicator.size(environment);
      auto const send_count = send_buffer.count() / size;
      assert(send_count * size == send_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter_c(
            send_buffer.data(), send_count.mpi_count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter(
            send_buffer.data(), send_count, send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::scatter", environment);
    }

    template <typename ReceiveValue>
    inline void scatter(
      MPI_Request& mpi_request,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) == root)
        throw ::yampi::nonroot_call_on_root_error("yampi::immediate_request_detail::scatter");

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter_c(
            nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter(
            nullptr, 0, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::scatter", environment);
    }

    template <typename Value>
    inline void scatter_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) == root)
        throw ::yampi::root_call_on_nonroot_error("yampi::immediate_request_detail::scatter_in_place");

      auto const size = communicator.size(environment);
      auto const send_count = send_buffer.count() / size;
      assert(send_count * size == send_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter_c(
            send_buffer.data(), send_count.mpi_count(), send_buffer.datatype().mpi_datatype(),
            MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter(
            send_buffer.data(), send_count, send_buffer.datatype().mpi_datatype(),
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::scatter_in_place", environment);
    }

    template <typename ReceiveValue>
    inline void scatter(
      MPI_Request& mpi_request,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter_c(
            nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter(
            nullptr, 0, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::scatter", environment);
    }

    template <typename SendValue>
    inline void scatter(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      auto const remote_size = communicator.remote_size(environment);
      auto const send_count = send_buffer.count() / remote_size;
      assert(send_count * remote_size == send_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter_c(
            send_buffer.data(), send_count.mpi_count(), send_buffer.datatype().mpi_datatype(),
            nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
            MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter(
            send_buffer.data(), send_count, send_buffer.datatype().mpi_datatype(),
            nullptr, 0, MPI_DATATYPE_NULL,
            MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::scatter", environment);
    }

    inline void scatter(
      MPI_Request& mpi_request,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter_c(
            nullptr, MPI_Count{0}, MPI_DATATYPE_NULL, nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
            MPI_PROC_NULL, communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatter(
            nullptr, 0, MPI_DATATYPE_NULL, nullptr, 0, MPI_DATATYPE_NULL,
            MPI_PROC_NULL, communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::scatter", environment);
    }

    // noncontiguous_scatter
    template <typename SendValue, typename ReceiveValue>
    inline void noncontiguous_scatter(
      MPI_Request& mpi_request,
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatterv_c(
            send_buffer.data(),
            reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
            send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatterv(
            send_buffer.data(), send_buffer.count_first(),
            send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_scatter", environment);
    }

    template <typename ReceiveValue>
    inline void noncontiguous_scatter(
      MPI_Request& mpi_request,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) == root)
        throw ::yampi::nonroot_call_on_root_error("yampi::immediate_request_detail::noncontiguous_scatter");

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatterv_c(
            nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatterv(
            nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_scatter", environment);
    }

    template <typename Value>
    inline void noncontiguous_scatter_in_place(
      MPI_Request& mpi_request,
      ::yampi::noncontiguous_buffer<Value, false> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) == root)
        throw ::yampi::root_call_on_nonroot_error("yampi::immediate_request_detail::noncontiguous_scatter_in_place");

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatterv_c(
            send_buffer.data(),
            reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
            send_buffer.datatype().mpi_datatype(),
            MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatterv(
            send_buffer.data(), send_buffer.count_first(),
            send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_scatter_in_place", environment);
    }

    template <typename ReceiveValue>
    inline void noncontiguous_scatter(
      MPI_Request& mpi_request,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatterv_c(
            nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatterv(
            nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_scatter", environment);
    }

    template <typename SendValue>
    inline void noncontiguous_scatter(
      MPI_Request& mpi_request,
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatterv_c(
            send_buffer.data(),
            reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
            send_buffer.datatype().mpi_datatype(),
            nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
            MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatterv(
            send_buffer.data(), send_buffer.count_first(),
            send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
            nullptr, 0, MPI_DATATYPE_NULL,
            MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_scatter", environment);
    }

    inline void noncontiguous_scatter(
      MPI_Request& mpi_request,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatterv_c(
            nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
            MPI_PROC_NULL, communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscatterv(
            nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, nullptr, 0, MPI_DATATYPE_NULL,
            MPI_PROC_NULL, communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_scatter", environment);
    }

    // all_gather
    template <typename SendValue, typename ContiguousIterator>
    inline void all_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.size(environment) <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * communicator.size(environment) <= send_buffer.data());
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgather_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgather(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::all_gather", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    inline void all_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
#   endif // MPI_VERSION >= 4

      auto const size = communicator.size(environment);
      auto const receive_count = receive_buffer.count() / size;
      assert(receive_count * size == receive_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgather_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgather(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::all_gather", environment);
    }

    template <typename Value>
    inline void all_gather_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      auto const size = communicator.size(environment);
      auto const receive_count = receive_buffer.count() / size;
      assert(receive_count * size == receive_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgather_c(
            MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgather(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::all_gather_in_place", environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    inline void all_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.remote_size(environment) <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * communicator.remote_size(environment) <= send_buffer.data());
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgather_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgather(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::all_gather", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    inline void all_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
#   endif // MPI_VERSION >= 4

      auto const remote_size = communicator.remote_size(environment);
      auto const receive_count = receive_buffer.count() / remote_size;
      assert(receive_count * remote_size == receive_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgather_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgather(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::all_gather", environment);
    }

    // neighbor all_gather
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    inline void all_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * topology.num_neighbors(environment) <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * topology.num_neighbors(environment) <= send_buffer.data());
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ineighbor_allgather_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ineighbor_allgather(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::all_gather", environment);
    }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    inline void all_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
#   endif // MPI_VERSION >= 4

      auto const num_neighbors = topology.num_neighbors(environment);
      auto const receive_count = receive_buffer.count() / num_neighbors;
      assert(receive_count * num_neighbors == receive_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ineighbor_allgather_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ineighbor_allgather(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::all_gather", environment);
    }

    // noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue>
    inline void noncontiguous_all_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgatherv_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(),
            reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
            receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgatherv(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count_first(),
            receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_all_gather", environment);
    }

    template <typename Value>
    inline void noncontiguous_all_gather_in_place(
      MPI_Request& mpi_request,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgatherv_c(
            MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
            receive_buffer.data(),
            reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
            receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallgatherv(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count_first(),
            receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_all_gather_in_place", environment);
    }

    // neighbor noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue, typename Topology>
    inline void noncontiguous_all_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ineighbor_allgatherv_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(),
            reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
            receive_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ineighbor_allgatherv(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count_first(),
            receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_all_gather", environment);
    }

    // complete_exchange
    template <typename SendValue, typename ContiguousIterator>
    inline void complete_exchange(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.size(environment) <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * communicator.size(environment) <= send_buffer.data());
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoall_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoall(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::complete_exchange", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    inline void complete_exchange(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
#   endif // MPI_VERSION >= 4

      auto const size = communicator.size(environment);
      auto const receive_count = receive_buffer.count() / size;
      assert(receive_count * size == receive_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoall_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoall(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::complete_exchange", environment);
    }

    template <typename Value>
    inline void complete_exchange_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      auto const size = communicator.size(environment);
      auto const receive_count = receive_buffer.count() / size;
      assert(receive_count * size == receive_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoall_c(
            MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoall(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::complete_exchange_in_place", environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    inline void complete_exchange(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.remote_size(environment) <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * communicator.remote_size(environment) <= send_buffer.data());
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoall_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoall(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::complete_exchange", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    inline void complete_exchange(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
#   endif // MPI_VERSION >= 4

      auto const remote_size = communicator.remote_size(environment);
      auto const receive_count = receive_buffer.count() / remote_size;
      assert(receive_count * remote_size == receive_buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoall_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoall(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::complete_exchange", environment);
    }

    // neighbor complete_exchange
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    inline void complete_exchange(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * topology.num_neighbors(environment) <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * topology.num_neighbors(environment) <= send_buffer.data());
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ineighbor_alltoall_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ineighbor_alltoall(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::complete_exchange", environment);
    }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    inline void complete_exchange(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
#   endif // MPI_VERSION >= 4

      auto const num_neighbors = topology.num_neighbors(environment);
      auto const receive_count = receive_buffer.count() / num_neighbors;
      assert(receive_count * num_neighbors == receive_buffer.count());

#   if MPI_VERSION >= 4
      int const error_code
        = MPI_Ineighbor_alltoall_c(
            send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      int const error_code
        = MPI_Ineighbor_alltoall(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::complete_exchange", environment);
    }

    // noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue>
    inline void noncontiguous_complete_exchange(
      MPI_Request& mpi_request,
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoallv_c(
            send_buffer.data(),
            reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
            send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(),
            reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
            receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoallv(
            send_buffer.data(), send_buffer.count_first(),
            send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count_first(),
            receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_complete_exchange", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    inline void noncontiguous_complete_exchange(
      MPI_Request& mpi_request,
      ::yampi::noncontiguous_buffer<SendValue, true> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, true> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoallw_c(
            send_buffer.data(),
            reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(send_buffer.byte_displacement_first()),
            reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
            receive_buffer.data(),
            reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
            reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoallw(
            send_buffer.data(), send_buffer.count_first(),
            send_buffer.displacement_first(), reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
            receive_buffer.data(), receive_buffer.count_first(),
            receive_buffer.displacement_first(), reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_complete_exchange", environment);
    }

    template <typename Value>
    inline void noncontiguous_complete_exchange_in_place(
      MPI_Request& mpi_request,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoallv_c(
            MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
            receive_buffer.data(),
            reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
            receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoallv(
            MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count_first(),
            receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_complete_exchange_in_place", environment);
    }

    template <typename Value>
    inline void noncontiguous_complete_exchange_in_place(
      MPI_Request& mpi_request,
      ::yampi::noncontiguous_buffer<Value, true> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoallw_c(
            MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
            receive_buffer.data(),
            reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
            reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ialltoallw(
            MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
            receive_buffer.data(), receive_buffer.count_first(),
            receive_buffer.displacement_first(), reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
            communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_complete_exchange_in_place", environment);
    }

    // neighbor noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, typename Topology>
    inline void noncontiguous_complete_exchange(
      MPI_Request& mpi_request,
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      int const error_code
        = MPI_Ineighbor_alltoallv_c(
            send_buffer.data(),
            reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
            send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(),
            reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
            receive_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      int const error_code
        = MPI_Ineighbor_alltoallv(
            send_buffer.data(), send_buffer.count_first(),
            send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count_first(),
            receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_complete_exchange", environment);
    }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    inline void noncontiguous_complete_exchange(
      MPI_Request& mpi_request,
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      int const error_code
        = MPI_Ineighbor_alltoallw_c(
            send_buffer.data(),
            reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(send_buffer.byte_displacement_first()),
            reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
            receive_buffer.data(),
            reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
            reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
            reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      int const error_code
        = MPI_Ineighbor_alltoallw(
            send_buffer.data(), send_buffer.count_first(),
            send_buffer.displacement_first(), reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
            receive_buffer.data(), receive_buffer.count_first(),
            receive_buffer.displacement_first(), reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
            topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::noncontiguous_complete_exchange", environment);
    }

    // reduce
    template <typename SendValue, typename ContiguousIterator>
    inline void reduce(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() <= send_buffer.data());
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce_c(
            send_buffer.data(), std::addressof(*first),
            send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
            std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce(
            send_buffer.data(), std::addressof(*first),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
            std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::reduce", environment);
    }

    template <typename SendValue>
    inline void reduce(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) == root)
        throw ::yampi::nonroot_call_on_root_error("yampi::immediate_request_detail::reduce");

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce_c(
            send_buffer.data(), nullptr,
            send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
            std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce(
            send_buffer.data(), nullptr,
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
            std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::reduce", environment);
    }

    template <typename Value>
    inline void reduce_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
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
#   else // MPI_VERSION >= 4
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
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::reduce_in_place", environment);
    }

    template <typename SendValue>
    inline void reduce(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce_c(
            send_buffer.data(), nullptr,
            send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
            std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce(
            send_buffer.data(), nullptr,
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
            std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::reduce", environment);
    }

    template <typename ReceiveValue>
    inline void reduce(
      MPI_Request& mpi_request,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce_c(
            nullptr, receive_buffer.data(),
            receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), MPI_ROOT, communicator.mpi_comm(),
            std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce(
            nullptr, receive_buffer.data(),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), MPI_ROOT, communicator.mpi_comm(),
            std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::reduce", environment);
    }

    inline void reduce(
      MPI_Request& mpi_request,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce_c(
            nullptr, nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
            MPI_OP_NULL, MPI_PROC_NULL, communicator.mpi_comm(),
            std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce(
            nullptr, nullptr, 0, MPI_DATATYPE_NULL,
            MPI_OP_NULL, MPI_PROC_NULL, communicator.mpi_comm(),
            std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::reduce", environment);
    }

    // all_reduce
    template <typename SendValue, typename ContiguousIterator>
    inline void all_reduce(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() <= send_buffer.data());
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallreduce_c(
            send_buffer.data(), std::addressof(*first),
            send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallreduce(
            send_buffer.data(), std::addressof(*first),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::all_reduce", environment);
    }

    template <typename Value>
    inline void all_reduce_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallreduce_c(
            MPI_IN_PLACE,
            buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iallreduce(
            MPI_IN_PLACE,
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::all_reduce_in_place", environment);
    }

    // reduce_scatter
    template <typename SendValue, typename ContiguousIterator>
    inline void reduce_scatter(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to Value");
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + 1 <= send_buffer.data());
      assert(send_buffer.count().mpi_count() == communicator.size(environment));
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + 1 <= send_buffer.data());
      assert(send_buffer.count() == communicator.size(environment));
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce_scatter_block_c(
            send_buffer.data(), std::addressof(*first), MPI_Count{1}, send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce_scatter_block(
            send_buffer.data(), std::addressof(*first), 1, send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::reduce_scatter", environment);
    }

    template <typename Value>
    inline void reduce_scatter(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
      assert(send_buffer.count().mpi_count() == communicator.size(environment) * receive_buffer.count().mpi_count());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
      assert(send_buffer.count() == communicator.size(environment) * receive_buffer.count());
#   endif // MPI_VERSION >= 4
      assert(send_buffer.datatype() == receive_buffer.datatype());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce_scatter_block_c(
            send_buffer.data(), receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce_scatter_block(
            send_buffer.data(), receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::reduce_scatter", environment);
    }

    template <typename Value>
    inline void reduce_scatter_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      auto const size = communicator.size(environment);
      auto const receive_count = buffer.count() / size;
      assert(receive_count * size == buffer.count());

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce_scatter_block_c(
            MPI_IN_PLACE, buffer.data(), receive_count.mpi_count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Ireduce_scatter_block(
            MPI_IN_PLACE, buffer.data(), receive_count, buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::reduce_scatter_in_place", environment);
    }

    // inclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    inline void inclusive_scan(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() <= send_buffer.data());
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscan_c(
            send_buffer.data(), std::addressof(*first),
            send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscan(
            send_buffer.data(), std::addressof(*first),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::inclusive_scan", environment);
    }

    template <typename Value>
    inline void inclusive_scan_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscan_c(
            MPI_IN_PLACE, buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iscan(
            MPI_IN_PLACE, buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::inclusive_scan_in_place", environment);
    }

    // exclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    inline void exclusive_scan(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (std::is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");
#   if MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() <= send_buffer.data());
#   else // MPI_VERSION >= 4
      assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() <= send_buffer.data());
#   endif // MPI_VERSION >= 4

#   if MPI_VERSION >= 4
      auto const error_code
        = MPI_Iexscan_c(
            send_buffer.data(), std::addressof(*first),
            send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      auto const error_code
        = MPI_Iexscan(
            send_buffer.data(), std::addressof(*first),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::exclusive_scan", environment);
    }

    template <typename Value>
    inline void exclusive_scan_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
#   if MPI_VERSION >= 4
      int const error_code
        = MPI_Iexscan_c(
            MPI_IN_PLACE, buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
      int const error_code
        = MPI_Iexscan(
            MPI_IN_PLACE, buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::exclusive_scan_in_place", environment);
    }

    // duplicate communicator
    inline void duplicate_communicator(
      MPI_Request& mpi_request,
      ::yampi::communicator_base const& old_communicator, ::yampi::communicator_base& new_communicator,
      ::yampi::environment const& environment)
    {
      MPI_Comm mpi_comm;
      auto const error_code
        = MPI_Comm_idup(old_communicator.mpi_comm(), std::addressof(mpi_comm), std::addressof(mpi_request));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::duplicate_communicator", environment);

      new_communicator.reset(mpi_comm, environment);
    }
#   if MPI_VERSION >= 4

    inline void duplicate_communicator(
      MPI_Request& mpi_request,
      ::yampi::communicator_base const& old_communicator, ::yampi::information const& information,
      ::yampi::communicator_base& new_communicator, ::yampi::environment const& environment)
    {
      MPI_Comm mpi_comm;
      auto const error_code
        = MPI_Comm_idup_with_info(old_communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_comm), std::addressof(mpi_request));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::immediate_request_detail::duplicate_communicator", environment);

      new_communicator.reset(mpi_comm, environment);
    }
#   endif // MPI_VERSION >= 4
# endif // MPI_VERSION >= 3
  }

# if MPI_VERSION >= 4
  struct request_send_receive_t { };
# endif // MPI_VERSION >= 4
  struct request_barrier_t { };
  struct request_broadcast_t { };
  struct request_gather_t { };
  struct request_noncontiguous_gather_t { };
  struct request_scatter_t { };
  struct request_noncontiguous_scatter_t { };
  struct request_all_gather_t { };
  struct request_noncontiguous_all_gather_t { };
  struct request_complete_exchange_t { };
  struct request_noncontiguous_complete_exchange_t { };
  struct request_reduce_t { };
  struct request_all_reduce_t { };
  struct request_reduce_scatter_t { };
  struct request_inclusive_scan_t { };
  struct request_exclusive_scan_t { };
  struct request_duplicate_communicator_t { };

# if __cplusplus >= 201703L
#   if MPI_VERSION >= 4
  inline constexpr ::yampi::request_send_receive_t request_send_receive{};
#   endif // MPI_VERSION >= 4
  inline constexpr ::yampi::request_barrier_t request_barrier{};
  inline constexpr ::yampi::request_broadcast_t request_broadcast{};
  inline constexpr ::yampi::request_gather_t request_gather{};
  inline constexpr ::yampi::request_noncontiguous_gather_t request_noncontiguous_gather{};
  inline constexpr ::yampi::request_scatter_t request_scatter{};
  inline constexpr ::yampi::request_noncontiguous_scatter_t request_noncontiguous_scatter{};
  inline constexpr ::yampi::request_all_gather_t request_all_gather{};
  inline constexpr ::yampi::request_noncontiguous_all_gather_t request_noncontiguous_all_gather{};
  inline constexpr ::yampi::request_complete_exchange_t request_complete_exchange{};
  inline constexpr ::yampi::request_noncontiguous_complete_exchange_t request_noncontiguous_complete_exchange{};
  inline constexpr ::yampi::request_reduce_t request_reduce{};
  inline constexpr ::yampi::request_all_reduce_t request_all_reduce{};
  inline constexpr ::yampi::request_reduce_scatter_t request_reduce_scatter{};
  inline constexpr ::yampi::request_inclusive_scan_t request_inclusive_scan{};
  inline constexpr ::yampi::request_exclusive_scan_t request_exclusive_scan{};
# else
#   if MPI_VERSION >= 4
  constexpr ::yampi::request_send_receive_t request_send_receive{};
#   endif // MPI_VERSION >= 4
  constexpr ::yampi::request_barrier_t request_barrier{};
  constexpr ::yampi::request_broadcast_t request_broadcast{};
  constexpr ::yampi::request_gather_t request_gather{};
  constexpr ::yampi::request_noncontiguous_gather_t request_noncontiguous_gather{};
  constexpr ::yampi::request_scatter_t request_scatter{};
  constexpr ::yampi::request_noncontiguous_scatter_t request_noncontiguous_scatter{};
  constexpr ::yampi::request_all_gather_t request_all_gather{};
  constexpr ::yampi::request_noncontiguous_all_gather_t request_noncontiguous_all_gather{};
  constexpr ::yampi::request_complete_exchange_t request_complete_exchange{};
  constexpr ::yampi::request_noncontiguous_complete_exchange_t request_noncontiguous_complete_exchange{};
  constexpr ::yampi::request_reduce_t request_reduce{};
  constexpr ::yampi::request_all_reduce_t request_all_reduce{};
  constexpr ::yampi::request_reduce_scatter_t request_reduce_scatter{};
  constexpr ::yampi::request_inclusive_scan_t request_inclusive_scan{};
  constexpr ::yampi::request_exclusive_scan_t request_exclusive_scan{};
# endif

  class immediate_request_ref;
  class immediate_request_cref;

  class immediate_request
    : public ::yampi::request_base
  {
    typedef ::yampi::request_base base_type;
    friend class ::yampi::immediate_request_ref;

   public:
    typedef ::yampi::immediate_request_ref reference_type;
    typedef ::yampi::immediate_request_cref const_reference_type;

    immediate_request() noexcept(std::is_nothrow_default_constructible<base_type>::value)
      : base_type{}
    { }

    immediate_request(immediate_request const&) = delete;
    immediate_request& operator=(immediate_request const&) = delete;
    immediate_request(immediate_request&&) = default;
    immediate_request& operator=(immediate_request&&) = default;
    ~immediate_request() noexcept = default;

    using base_type::base_type;

    // send
    template <typename Value>
    immediate_request(
      ::yampi::request_send_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_standard_send_request(buffer, destination, tag, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_send_t const,
      ::yampi::mode::standard_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_standard_send_request(buffer, destination, tag, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_send_t const,
      ::yampi::mode::buffered_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_buffered_send_request(buffer, destination, tag, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_send_t const,
      ::yampi::mode::synchronous_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_synchronous_send_request(buffer, destination, tag, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_send_t const,
      ::yampi::mode::ready_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_ready_send_request(buffer, destination, tag, communicator, environment)}
    { }

    // receive
    template <typename Value>
    immediate_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_receive_request(buffer, source, tag, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_receive_request(buffer, source, ::yampi::any_tag, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_receive_request(buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment)}
    { }
# if MPI_VERSION >= 3

    template <typename Value>
    immediate_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::message& message,
      ::yampi::environment const& environment)
      : base_type{make_receive_request(buffer, message, environment)}
    { }
# endif // MPI_VERSION >= 3

# if MPI_VERSION >= 4
    // send_receive
    template <typename SendValue, typename ReceiveValue>
    immediate_request(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_send_receive_request(send_buffer, destination, send_tag, receive_buffer, source, receive_tag, communicator, environment)}
    { }

    template <typename SendValue, typename ReceiveValue>
    immediate_request(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_send_receive_request(send_buffer, destination, send_tag, receive_buffer, source, ::yampi::any_tag, communicator, environment)}
    { }

    template <typename SendValue, typename ReceiveValue>
    immediate_request(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_send_receive_request(send_buffer, destination, send_tag, receive_buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment)}
    { }

    // with replacement
    template <typename Value>
    immediate_request(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_send_receive_request(buffer, destination, send_tag, source, receive_tag, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_send_receive_request(buffer, destination, send_tag, source, ::yampi::any_tag, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_send_receive_request(buffer, destination, send_tag, ::yampi::any_source, ::yampi::any_tag, communicator, environment)}
    { }
# endif // MPI_VERSION >= 4

# if MPI_VERSION >= 3
    // Nonblocking collective operations
    // barrier
    template <typename Value>
    immediate_request(
      ::yampi::request_barrier_t const,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_barrier_request(communicator, environment)}
    { }

    // broadcast
    template <typename Value>
    immediate_request(
      ::yampi::request_broadcast_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const root,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_broadcast_request(buffer, root, communicator, environment)}
    { }

    template <typename SendValue>
    immediate_request(
      ::yampi::request_broadcast_t const,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_broadcast_request(send_buffer, communicator, environment)}
    { }

    immediate_request(
      ::yampi::request_broadcast_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_broadcast_request(communicator, environment)}
    { }

    // gather
    template <typename SendValue, typename ContiguousIterator>
    immediate_request(
      ::yampi::request_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_gather_request(send_buffer, first, root, communicator, environment)}
    { }

    template <typename SendValue, typename ReceiveValue>
    immediate_request(
      ::yampi::request_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_gather_request(send_buffer, receive_buffer, root, communicator, environment)}
    { }

    template <typename SendValue>
    immediate_request(
      ::yampi::request_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_gather_request(send_buffer, root, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_gather_t const,
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_gather_in_place_request(receive_buffer, root, communicator, environment)}
    { }

    template <typename SendValue>
    immediate_request(
      ::yampi::request_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_gather_request(send_buffer, root, communicator, environment)}
    { }

    template <typename ReceiveValue>
    immediate_request(
      ::yampi::request_gather_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_gather_request(receive_buffer, communicator, environment)}
    { }

    immediate_request(
      ::yampi::request_gather_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_gather_request(communicator, environment)}
    { }

    // noncontiguous_gather
    template <typename SendValue, typename ReceiveValue>
    immediate_request(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_gather_request(send_buffer, receive_buffer, root, communicator, environment)}
    { }

    template <typename SendValue>
    immediate_request(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_gather_request(send_buffer, root, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::in_place_t const,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_gather_in_place_request(receive_buffer, root, communicator, environment)}
    { }

    template <typename SendValue>
    immediate_request(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_gather_request(send_buffer, root, communicator, environment)}
    { }

    template <typename ReceiveValue>
    immediate_request(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_gather_request(receive_buffer, communicator, environment)}
    { }

    immediate_request(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_gather_request(communicator, environment)}
    { }

    // scatter
    template <typename ContiguousIterator, typename ReceiveValue>
    immediate_request(
      ::yampi::request_scatter_t const,
      ContiguousIterator const first, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_scatter_request(first, receive_buffer, root, communicator, environment)}
    { }

    template <typename SendValue, typename ReceiveValue>
    immediate_request(
      ::yampi::request_scatter_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_scatter_request(send_buffer, receive_buffer, root, communicator, environment)}
    { }

    template <typename ReceiveValue>
    immediate_request(
      ::yampi::request_scatter_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_scatter_request(receive_buffer, root, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_scatter_t const,
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_scatter_in_place_request(send_buffer, root, communicator, environment)}
    { }

    template <typename ReceiveValue>
    immediate_request(
      ::yampi::request_scatter_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_scatter_request(receive_buffer, root, communicator, environment)}
    { }

    template <typename SendValue>
    immediate_request(
      ::yampi::request_scatter_t const,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_scatter_request(send_buffer, communicator, environment)}
    { }

    immediate_request(
      ::yampi::request_scatter_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_scatter_request(communicator, environment)}
    { }

    // noncontiguous_scatter
    template <typename SendValue, typename ReceiveValue>
    immediate_request(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_scatter_request(send_buffer, receive_buffer, root, communicator, environment)}
    { }

    template <typename ReceiveValue>
    immediate_request(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_scatter_request(receive_buffer, root, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::in_place_t const,
      ::yampi::noncontiguous_buffer<Value, false> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_scatter_in_place_request(send_buffer, root, communicator, environment)}
    { }

    template <typename ReceiveValue>
    immediate_request(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_scatter_request(receive_buffer, root, communicator, environment)}
    { }

    template <typename SendValue>
    immediate_request(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_scatter_request(send_buffer, communicator, environment)}
    { }

    immediate_request(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_scatter_request(communicator, environment)}
    { }

    // all_gather
    template <typename SendValue, typename ContiguousIterator>
    immediate_request(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_all_gather_request(send_buffer, first, communicator, environment)}
    { }

    template <typename SendValue, typename ReceiveValue>
    immediate_request(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_all_gather_request(send_buffer, receive_buffer, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_all_gather_t const,
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_all_gather_in_place_request(receive_buffer, communicator, environment)}
    { }

    template <typename SendValue, typename ContiguousIterator>
    immediate_request(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_all_gather_request(send_buffer, first, communicator, environment)}
    { }

    template <typename SendValue, typename ReceiveValue>
    immediate_request(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_all_gather_request(send_buffer, receive_buffer, communicator, environment)}
    { }

    // neighbor all_gather
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    immediate_request(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
      : base_type{make_all_gather_request(send_buffer, first, topology, environment)}
    { }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    immediate_request(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
      : base_type{make_all_gather_request(send_buffer, receive_buffer, topology, environment)}
    { }

    // noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue>
    immediate_request(
      ::yampi::request_noncontiguous_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_all_gather_request(send_buffer, receive_buffer, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_noncontiguous_all_gather_t const,
      ::yampi::in_place_t const,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_all_gather_in_place_request(receive_buffer, communicator, environment)}
    { }

    // neighbor noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue, typename Topology>
    immediate_request(
      ::yampi::request_noncontiguous_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_all_gather_request(send_buffer, receive_buffer, topology, environment)}
    { }

    // complete_exchange
    template <typename SendValue, typename ContiguousIterator>
    immediate_request(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_complete_exchange_request(send_buffer, first, communicator, environment)}
    { }

    template <typename SendValue, typename ReceiveValue>
    immediate_request(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_complete_exchange_request(send_buffer, receive_buffer, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_complete_exchange_t const,
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_complete_exchange_in_place_request(receive_buffer, communicator, environment)}
    { }

    template <typename SendValue, typename ContiguousIterator>
    immediate_request(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_complete_exchange_request(send_buffer, first, communicator, environment)}
    { }

    template <typename SendValue, typename ReceiveValue>
    immediate_request(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_complete_exchange_request(send_buffer, receive_buffer, communicator, environment)}
    { }

    // neighbor complete_exchange
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    immediate_request(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
      : base_type{make_complete_exchange_request(send_buffer, first, topology, environment)}
    { }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    immediate_request(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
      : base_type{make_complete_exchange_request(send_buffer, receive_buffer, topology, environment)}
    { }

    // noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, bool enables_multiple_datatypes>
    immediate_request(
      ::yampi::request_noncontiguous_complete_exchange_t const,
      ::yampi::noncontiguous_buffer<SendValue, enables_multiple_datatypes> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, enables_multiple_datatypes> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_complete_exchange_request(send_buffer, receive_buffer, communicator, environment)}
    { }

    template <typename Value, bool enables_multiple_datatypes>
    immediate_request(
      ::yampi::request_noncontiguous_complete_exchange_t const,
      ::yampi::in_place_t const,
      ::yampi::noncontiguous_buffer<Value, enables_multiple_datatypes> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_complete_exchange_in_place_request(receive_buffer, communicator, environment)}
    { }

    // neighbor noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, bool enables_multiple_datatypes, typename Topology>
    immediate_request(
      ::yampi::request_noncontiguous_complete_exchange_t const,
      ::yampi::noncontiguous_buffer<SendValue, enables_multiple_datatypes> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, enables_multiple_datatypes> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
      : base_type{make_noncontiguous_complete_exchange_request(send_buffer, receive_buffer, topology, environment)}
    { }

    // reduce
    template <typename SendValue, typename ContiguousIterator>
    immediate_request(
      ::yampi::request_reduce_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_reduce_request(send_buffer, first, operation, root, communicator, environment)}
    { }

    template <typename SendValue>
    immediate_request(
      ::yampi::request_reduce_t const,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_reduce_request(send_buffer, operation, root, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_reduce_t const,
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_reduce_in_place_request(buffer, operation, root, communicator, environment)}
    { }

    template <typename SendValue>
    immediate_request(
      ::yampi::request_reduce_t const,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_reduce_request(send_buffer, operation, root, communicator, environment)}
    { }

    template <typename ReceiveValue>
    immediate_request(
      ::yampi::request_reduce_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_reduce_request(receive_buffer, operation, communicator, environment)}
    { }

    immediate_request(
      ::yampi::request_reduce_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_reduce_request(communicator, environment)}
    { }

    // all_reduce
    template <typename SendValue, typename ContiguousIterator>
    immediate_request(
      ::yampi::request_all_reduce_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_all_reduce_request(send_buffer, first, operation, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_all_reduce_t const,
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_all_reduce_in_place_request(buffer, operation, communicator, environment)}
    { }

    // reduce_scatter
    template <typename SendValue, typename ContiguousIterator>
    immediate_request(
      ::yampi::request_reduce_scatter_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_reduce_scatter_request(send_buffer, first, operation, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_reduce_scatter_t const,
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_reduce_scatter_request(send_buffer, receive_buffer, operation, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_reduce_scatter_t const,
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_reduce_scatter_in_place_request(buffer, operation, communicator, environment)}
    { }

    // inclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    immediate_request(
      ::yampi::request_inclusive_scan_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_inclusive_scan_request(send_buffer, first, operation, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_inclusive_scan_t const,
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_inclusive_scan_in_place_request(buffer, operation, communicator, environment)}
    { }

    // exclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    immediate_request(
      ::yampi::request_exclusive_scan_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_exclusive_scan_request(send_buffer, first, operation, communicator, environment)}
    { }

    template <typename Value>
    immediate_request(
      ::yampi::request_exclusive_scan_t const,
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type{make_exclusive_scan_in_place_request(buffer, operation, communicator, environment)}
    { }

    // duplicate communicator
    template <typename Value>
    immediate_request(
      ::yampi::request_duplicate_communicator_t const,
      ::yampi::communicator_base const& old_communicator, ::yampi::communicator_base& new_communicator,
      ::yampi::environment const& environment)
      : base_type{make_duplicate_communicator_request(old_communicator, new_communicator, environment)}
    { }
#   if MPI_VERSION >= 4

    template <typename Value>
    immediate_request(
      ::yampi::request_duplicate_communicator_t const,
      ::yampi::communicator_base const& old_communicator, ::yampi::information const& information,
      ::yampi::communicator_base& new_communicator, ::yampi::environment const& environment)
      : base_type{make_duplicate_communicator_request(old_communicator, information, new_communicator, environment)}
    { }
#   endif // MPI_VERSION >= 4
# endif // MPI_VERSION >= 3

   private:
    // send
    template <typename Value>
    MPI_Request make_standard_send_request(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::standard_send(result, buffer, destination, tag, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_buffered_send_request(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::buffered_send(result, buffer, destination, tag, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_synchronous_send_request(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::synchronous_send(result, buffer, destination, tag, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_ready_send_request(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::ready_send(result, buffer, destination, tag, communicator, environment);
      return result;
    }

    // receive
    template <typename Value>
    MPI_Request make_receive_request(
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::receive(result, buffer, source, tag, communicator, environment);
      return result;
    }
# if MPI_VERSION >= 3

    template <typename Value>
    MPI_Request make_receive_request(
      ::yampi::buffer<Value> buffer, ::yampi::message& message,
      ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::receive(result, buffer, message, environment);
      return result;
    }
# endif // MPI_VERSION >= 3

# if MPI_VERSION >= 4
    // send_receive
    template <typename SendValue, typename ReceiveValue>
    MPI_Request make_send_receive_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::send_receive(result, send_buffer, destination, send_tag, receive_buffer, source, receive_tag, communicator, environment);
      return result;
    }

    // with replacement
    template <typename Value>
    MPI_Request make_send_receive_request(
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::send_receive(result, buffer, destination, send_tag, source, receive_tag, communicator, environment);
      return result;
    }
# endif // MPI_VERSION >= 4

# if MPI_VERSION >= 3
    // Nonblocking collective operations
    // barrier
    MPI_Request make_barrier_request(::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::barrier(result, communicator, environment);
      return result;
    }

    // broadcast
    template <typename Value>
    MPI_Request make_broadcast_request(
      ::yampi::buffer<Value> buffer, ::yampi::rank const root,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::broadcast(result, buffer, root, communicator, environment);
      return result;
    }

    template <typename SendValue>
    MPI_Request make_broadcast_request(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::broadcast(result, send_buffer, communicator, environment);
      return result;
    }

    MPI_Request make_broadcast_request(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::broadcast(result, communicator, environment);
      return result;
    }

    // gather
    template <typename SendValue, typename ContiguousIterator>
    MPI_Request make_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::gather(result, send_buffer, first, root, communicator, environment);
      return result;
    }

    template <typename SendValue, typename ReceiveValue>
    MPI_Request make_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::gather(result, send_buffer, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename SendValue>
    MPI_Request make_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::gather(result, send_buffer, root, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_gather_in_place_request(
      ::yampi::buffer<Value> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::gather_in_place(result, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename SendValue>
    MPI_Request make_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::gather(result, send_buffer, root, communicator, environment);
      return result;
    }

    template <typename ReceiveValue>
    MPI_Request make_gather_request(
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::gather(result, receive_buffer, communicator, environment);
      return result;
    }

    MPI_Request make_gather_request(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::gather(result, communicator, environment);
      return result;
    }

    // noncontiguous_gather
    template <typename SendValue, typename ReceiveValue>
    MPI_Request make_noncontiguous_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_gather(result, send_buffer, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename SendValue>
    MPI_Request make_noncontiguous_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_gather(result, send_buffer, root, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_noncontiguous_gather_in_place_request(
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_gather_in_place(result, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename SendValue>
    MPI_Request make_noncontiguous_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_gather(result, send_buffer, root, communicator, environment);
      return result;
    }

    template <typename ReceiveValue>
    MPI_Request make_noncontiguous_gather_request(
      ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_gather(result, receive_buffer, communicator, environment);
      return result;
    }

    MPI_Request make_noncontiguous_gather_request(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_gather(result, communicator, environment);
      return result;
    }

    // scatter
    template <typename ContiguousIterator, typename ReceiveValue>
    MPI_Request make_scatter_request(
      ContiguousIterator const first, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::scatter(result, first, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename SendValue, typename ReceiveValue>
    MPI_Request make_scatter_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::scatter(result, send_buffer, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename ReceiveValue>
    MPI_Request make_scatter_request(
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::scatter(result, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_scatter_in_place_request(
      ::yampi::buffer<Value> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::scatter_in_place(result, send_buffer, root, communicator, environment);
      return result;
    }

    template <typename ReceiveValue>
    MPI_Request make_scatter_request(
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::scatter(result, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename SendValue>
    MPI_Request make_scatter_request(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::scatter(result, send_buffer, communicator, environment);
      return result;
    }

    MPI_Request make_scatter_request(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::scatter(result, communicator, environment);
      return result;
    }

    // noncontiguous_scatter
    template <typename SendValue, typename ReceiveValue>
    MPI_Request make_noncontiguous_scatter_request(
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_scatter(result, send_buffer, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename ReceiveValue>
    MPI_Request make_noncontiguous_scatter_request(
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_scatter(result, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_noncontiguous_scatter_in_place_request(
      ::yampi::noncontiguous_buffer<Value, false> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_scatter_in_place(result, send_buffer, root, communicator, environment);
      return result;
    }

    template <typename ReceiveValue>
    MPI_Request make_noncontiguous_scatter_request(
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_scatter(result, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename SendValue>
    MPI_Request make_noncontiguous_scatter_request(
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_scatter(result, send_buffer, communicator, environment);
      return result;
    }

    MPI_Request make_noncontiguous_scatter_request(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_scatter(result, communicator, environment);
      return result;
    }

    // all_gather
    template <typename SendValue, typename ContiguousIterator>
    MPI_Request make_all_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::all_gather(result, send_buffer, first, communicator, environment);
      return result;
    }

    template <typename SendValue, typename ReceiveValue>
    MPI_Request make_all_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::all_gather(result, send_buffer, receive_buffer, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_all_gather_in_place_request(
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::all_gather_in_place(result, receive_buffer, communicator, environment);
      return result;
    }

    template <typename SendValue, typename ContiguousIterator>
    MPI_Request make_all_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::all_gather(result, send_buffer, first, communicator, environment);
      return result;
    }

    template <typename SendValue, typename ReceiveValue>
    MPI_Request make_all_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::all_gather(result, send_buffer, receive_buffer, communicator, environment);
      return result;
    }

    // neighbor all_gather
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    MPI_Request make_all_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::all_gather(result, send_buffer, first, topology, environment);
      return result;
    }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    MPI_Request make_all_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::all_gather(result, send_buffer, receive_buffer, topology, environment);
      return result;
    }

    // noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue>
    MPI_Request make_noncontiguous_all_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_all_gather(result, send_buffer, receive_buffer, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_noncontiguous_all_gather_in_place_request(
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_all_gather_in_place(result, receive_buffer, communicator, environment);
      return result;
    }

    // neighbor noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue, typename Topology>
    MPI_Request make_noncontiguous_all_gather_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_all_gather(result, send_buffer, receive_buffer, topology, environment);
      return result;
    }

    // complete_exchange
    template <typename SendValue, typename ContiguousIterator>
    MPI_Request make_complete_exchange_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::complete_exchange(result, send_buffer, first, communicator, environment);
      return result;
    }

    template <typename SendValue, typename ReceiveValue>
    MPI_Request make_complete_exchange_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::complete_exchange(result, send_buffer, receive_buffer, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_complete_exchange_in_place_request(
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::complete_exchange_in_place(result, receive_buffer, communicator, environment);
      return result;
    }

    template <typename SendValue, typename ContiguousIterator>
    MPI_Request make_complete_exchange_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::complete_exchange(result, send_buffer, first, communicator, environment);
      return result;
    }

    template <typename SendValue, typename ReceiveValue>
    MPI_Request make_complete_exchange_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::complete_exchange(result, send_buffer, receive_buffer, communicator, environment);
      return result;
    }

    // neighbor complete_exchange
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    MPI_Request make_complete_exchange_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::complete_exchange(result, send_buffer, first, topology, environment);
      return result;
    }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    MPI_Request make_complete_exchange_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::complete_exchange(result, send_buffer, receive_buffer, topology, environment);
      return result;
    }

    // noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, bool enables_multiple_datatypes>
    MPI_Request make_noncontiguous_complete_exchange_request(
      ::yampi::noncontiguous_buffer<SendValue, enables_multiple_datatypes> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, enables_multiple_datatypes> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_complete_exchange(result, send_buffer, receive_buffer, communicator, environment);
      return result;
    }

    template <typename Value, bool enables_multiple_datatypes>
    MPI_Request make_noncontiguous_complete_exchange_in_place_request(
      ::yampi::noncontiguous_buffer<Value, enables_multiple_datatypes> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_complete_exchange_in_place(result, receive_buffer, communicator, environment);
      return result;
    }

    // neighbor noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, bool enables_multiple_datatypes, typename Topology>
    MPI_Request make_noncontiguous_complete_exchange_request(
      ::yampi::noncontiguous_buffer<SendValue, enables_multiple_datatypes> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, enables_multiple_datatypes> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::noncontiguous_complete_exchange(result, send_buffer, receive_buffer, topology, environment);
      return result;
    }

    // reduce
    template <typename SendValue, typename ContiguousIterator>
    MPI_Request make_reduce_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::reduce(result, send_buffer, first, operation, root, communicator, environment);
      return result;
    }

    template <typename SendValue>
    MPI_Request make_reduce_request(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::reduce(result, send_buffer, operation, root, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_reduce_in_place_request(
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::reduce_in_place(result, buffer, operation, root, communicator, environment);
      return result;
    }

    template <typename SendValue>
    MPI_Request make_reduce_request(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::reduce(result, send_buffer, operation, root, communicator, environment);
      return result;
    }

    template <typename ReceiveValue>
    MPI_Request make_reduce_request(
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::reduce(result, receive_buffer, operation, communicator, environment);
      return result;
    }

    MPI_Request make_reduce_request(
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::reduce(result, communicator, environment);
      return result;
    }

    // all_reduce
    template <typename SendValue, typename ContiguousIterator>
    MPI_Request make_all_reduce_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::all_reduce(result, send_buffer, first, operation, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_all_reduce_in_place_request(
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::all_reduce_in_place(result, buffer, operation, communicator, environment);
      return result;
    }

    // reduce_scatter
    template <typename SendValue, typename ContiguousIterator>
    MPI_Request make_reduce_scatter_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::reduce_scatter(result, send_buffer, first, operation, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_reduce_scatter_request(
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::reduce_scatter(result, send_buffer, receive_buffer, operation, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_reduce_scatter_in_place_request(
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::reduce_scatter_in_place(result, buffer, operation, communicator, environment);
      return result;
    }

    // inclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    MPI_Request make_inclusive_scan_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::inclusive_scan(result, send_buffer, first, operation, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_inclusive_scan_in_place_request(
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::inclusive_scan_in_place(result, buffer, operation, communicator, environment);
      return result;
    }

    // exclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    MPI_Request make_exclusive_scan_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::exclusive_scan(result, send_buffer, first, operation, communicator, environment);
      return result;
    }

    template <typename Value>
    MPI_Request make_exclusive_scan_in_place_request(
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::exclusive_scan_in_place(result, buffer, operation, communicator, environment);
      return result;
    }

    // duplicate communicator
    MPI_Request make_duplicate_communicator_request(
      ::yampi::communicator_base const& old_communicator, ::yampi::communicator_base& new_communicator,
      ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::duplicate_communicator(result, old_communicator, new_communicator, environment);
      return result;
    }
#   if MPI_VERSION >= 4

    MPI_Request make_duplicate_communicator_request(
      ::yampi::communicator_base const& old_communicator, ::yampi::information const& information,
      ::yampi::communicator_base& new_communicator, ::yampi::environment const& environment) const
    {
      MPI_Request result;
      ::yampi::immediate_request_detail::duplicate_communicator(result, old_communicator, information, new_communicator, environment);
      return result;
    }
#   endif // MPI_VERSION >= 4
# endif // MPI_VERSION >= 3

   public:
    using base_type::reset;

    // send
    template <typename Value>
    void reset(
      ::yampi::request_send_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send(buffer, destination, tag, communicator, environment);
    }

    template <typename Mode, typename Value>
    void reset(
      ::yampi::request_send_t const, Mode const mode,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send(mode, buffer, destination, tag, communicator, environment);
    }

    // receive
    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, communicator, environment);
    }
# if MPI_VERSION >= 3

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::message& message,
      ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, message, environment);
    }
# endif // MPI_VERSION >= 3

# if MPI_VERSION >= 4
    // send_receive
    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send_receive(send_buffer, destination, send_tag, receive_buffer, source, receive_tag, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send_receive(send_buffer, destination, send_tag, receive_buffer, source, ::yampi::any_tag, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send_receive(send_buffer, destination, send_tag, receive_buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment);
    }

    // with replacement
    template <typename Value>
    void reset(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send_receive(buffer, destination, send_tag, source, receive_tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send_receive(buffer, destination, send_tag, source, ::yampi::any_tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send_receive(buffer, destination, send_tag, ::yampi::any_source, ::yampi::any_tag, communicator, environment);
    }
# endif // MPI_VERSION >= 4

# if MPI_VERSION >= 3
    // Nonblocking collective operations
    // barrier
    void reset(
      ::yampi::request_barrier_t const,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      barrier(communicator, environment);
    }

    // broadcast
    template <typename Value>
    void reset(
      ::yampi::request_broadcast_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const root,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      broadcast(buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_broadcast_t const,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      broadcast(send_buffer, communicator, environment);
    }

    void reset(
      ::yampi::request_broadcast_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      broadcast(communicator, environment);
    }

    // gather
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(send_buffer, first, root, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(send_buffer, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(send_buffer, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_gather_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(in_place, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(send_buffer, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_gather_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(receive_buffer, communicator, environment);
    }

    void reset(
      ::yampi::request_gather_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(communicator, environment);
    }

    // noncontiguous_gather
    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_gather(send_buffer, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_gather(send_buffer, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_gather(in_place, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_gather(send_buffer, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_gather(receive_buffer, communicator, environment);
    }

    void reset(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_gather(communicator, environment);
    }

    // scatter
    template <typename ContiguousIterator, typename ReceiveValue>
    void reset(
      ::yampi::request_scatter_t const,
      ContiguousIterator const first, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(first, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_scatter_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(send_buffer, receive_buffer, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_scatter_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(receive_buffer, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_scatter_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(in_place, send_buffer, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_scatter_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_scatter_t const,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(send_buffer, communicator, environment);
    }

    void reset(
      ::yampi::request_scatter_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(communicator, environment);
    }

    // noncontiguous_scatter
    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_scatter(send_buffer, receive_buffer, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_scatter(receive_buffer, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::noncontiguous_buffer<Value, false> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_scatter(in_place, send_buffer, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_scatter(receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_scatter(send_buffer, communicator, environment);
    }

    void reset(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_scatter(communicator, environment);
    }

    // all_gather
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(send_buffer, first, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(send_buffer, receive_buffer, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(in_place, receive_buffer, communicator, environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(send_buffer, first, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(send_buffer, receive_buffer, communicator, environment);
    }

    // neighbor all_gather
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(send_buffer, first, topology, environment);
    }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(send_buffer, receive_buffer, topology, environment);
    }

    // noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_noncontiguous_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_all_gather(send_buffer, receive_buffer, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_noncontiguous_all_gather_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_all_gather(in_place, receive_buffer, communicator, environment);
    }

    // neighbor noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue, typename Topology>
    void reset(
      ::yampi::request_noncontiguous_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_all_gather(send_buffer, receive_buffer, topology, environment);
    }

    // complete_exchange
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, first, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, receive_buffer, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(in_place, receive_buffer, communicator, environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, first, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, receive_buffer, communicator, environment);
    }

    // neighbor complete_exchange
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, first, topology, environment);
    }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, receive_buffer, topology, environment);
    }

    // noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, bool enables_multiple_datatypes>
    void reset(
      ::yampi::request_noncontiguous_complete_exchange_t const,
      ::yampi::noncontiguous_buffer<SendValue, enables_multiple_datatypes> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, enables_multiple_datatypes> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_complete_exchange(send_buffer, receive_buffer, communicator, environment);
    }

    template <typename Value, bool enables_multiple_datatypes>
    void reset(
      ::yampi::request_noncontiguous_complete_exchange_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::noncontiguous_buffer<Value, enables_multiple_datatypes> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_complete_exchange(in_place, receive_buffer, communicator, environment);
    }

    // neighbor noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, bool enables_multiple_datatypes, typename Topology>
    void reset(
      ::yampi::request_noncontiguous_complete_exchange_t const,
      ::yampi::noncontiguous_buffer<SendValue, enables_multiple_datatypes> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, enables_multiple_datatypes> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_complete_exchange(send_buffer, receive_buffer, topology, environment);
    }

    // reduce
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_reduce_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(send_buffer, first, operation, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_reduce_t const,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(send_buffer, operation, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_reduce_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(in_place, buffer, operation, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_reduce_t const,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(send_buffer, operation, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_reduce_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(receive_buffer, operation, communicator, environment);
    }

    void reset(
      ::yampi::request_reduce_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(communicator, environment);
    }

    // all_reduce
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_all_reduce_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_reduce(send_buffer, first, operation, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_all_reduce_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_reduce(in_place, buffer, operation, communicator, environment);
    }

    // reduce_scatter
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_reduce_scatter_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce_scatter(send_buffer, first, operation, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_reduce_scatter_t const,
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce_scatter(send_buffer, receive_buffer, operation, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_reduce_scatter_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce_scatter(in_place, buffer, operation, communicator, environment);
    }

    // inclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_inclusive_scan_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      inclusive_scan(send_buffer, first, operation, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_inclusive_scan_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      inclusive_scan(in_place, buffer, operation, communicator, environment);
    }

    // exclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_exclusive_scan_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      exclusive_scan(send_buffer, first, operation, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_exclusive_scan_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      exclusive_scan(in_place, buffer, operation, communicator, environment);
    }

    // duplicate communicator
    template <typename Value>
    void reset(
      ::yampi::request_duplicate_communicator_t const,
      ::yampi::communicator_base const& old_communicator, ::yampi::communicator_base& new_communicator,
      ::yampi::environment const& environment)
    {
      free(environment);
      duplicate_communicator(old_communicator, new_communicator, environment);
    }
#   if MPI_VERSION >= 4

    template <typename Value>
    void reset(
      ::yampi::request_duplicate_communicator_t const,
      ::yampi::communicator_base const& old_communicator, ::yampi::information const& information,
      ::yampi::communicator_base& new_communicator, ::yampi::environment const& environment)
    {
      free(environment);
      duplicate_communicator(old_communicator, information, new_communicator, environment);
    }
#   endif // MPI_VERSION >= 4
# endif // MPI_VERSION >= 3

    // send
    template <typename Value>
    void send(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::standard_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::standard_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::standard_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::buffered_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::buffered_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::synchronous_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::synchronous_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::ready_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::ready_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    // receive
    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::receive(mpi_request_, buffer, source, tag, communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::receive(mpi_request_, buffer, source, ::yampi::any_tag, communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::receive(mpi_request_, buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment); }
# if MPI_VERSION >= 3

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer, ::yampi::message& message,
      ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::receive(mpi_request_, buffer, message, environment); }
# endif // MPI_VERSION >= 3

# if MPI_VERSION >= 4
    // send_receive
    template <typename SendValue, typename ReceiveValue>
    void send_receive(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::send_receive(mpi_request_, send_buffer, destination, send_tag, receive_buffer, source, receive_tag, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void send_receive(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::send_receive(mpi_request_, send_buffer, destination, send_tag, receive_buffer, source, ::yampi::any_tag, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void send_receive(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::send_receive(mpi_request_, send_buffer, destination, send_tag, receive_buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment); }

    // with replacement
    template <typename Value>
    void send_receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::send_receive(mpi_request_, buffer, destination, send_tag, source, receive_tag, communicator, environment); }

    template <typename Value>
    void send_receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::send_receive(mpi_request_, buffer, destination, send_tag, source, ::yampi::any_tag, communicator, environment); }

    template <typename Value>
    void send_receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::send_receive(mpi_request_, buffer, destination, send_tag, ::yampi::any_source, ::yampi::any_tag, communicator, environment); }
# endif // MPI_VERSION >= 4

# if MPI_VERSION >= 3
    // Nonblocking collective operations
    // barrier
    void barrier(::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::barrier(mpi_request_, communicator, environment); }

    // broadcast
    template <typename Value>
    void broadcast(
      ::yampi::buffer<Value> buffer, ::yampi::rank const root,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::broadcast(mpi_request_, buffer, root, communicator, environment); }

    template <typename SendValue>
    void broadcast(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::broadcast(mpi_request_, send_buffer, communicator, environment); }

    void broadcast(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::broadcast(mpi_request_, communicator, environment); }

    // gather
    template <typename SendValue, typename ContiguousIterator>
    void gather(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather(mpi_request_, send_buffer, first, root, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather(mpi_request_, send_buffer, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather(mpi_request_, send_buffer, root, communicator, environment); }

    template <typename Value>
    void gather(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather_in_place(mpi_request_, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather(mpi_request_, send_buffer, root, communicator, environment); }

    template <typename ReceiveValue>
    void gather(
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather(mpi_request_, receive_buffer, communicator, environment); }

    void gather(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather(mpi_request_, communicator, environment); }

    // noncontiguous_gather
    template <typename SendValue, typename ReceiveValue>
    void noncontiguous_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_gather(mpi_request_, send_buffer, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void noncontiguous_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_gather(mpi_request_, send_buffer, root, communicator, environment); }

    template <typename Value>
    void noncontiguous_gather(
      ::yampi::in_place_t const,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_gather_in_place(mpi_request_, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void noncontiguous_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_gather(mpi_request_, send_buffer, root, communicator, environment); }

    template <typename ReceiveValue>
    void noncontiguous_gather(
      ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_gather(mpi_request_, receive_buffer, communicator, environment); }

    void noncontiguous_gather(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_gather(mpi_request_, communicator, environment); }

    // scatter
    template <typename ContiguousIterator, typename ReceiveValue>
    void scatter(
      ContiguousIterator const first, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter(mpi_request_, first, receive_buffer, root, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void scatter(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter(mpi_request_, send_buffer, receive_buffer, root, communicator, environment); }

    template <typename ReceiveValue>
    void scatter(
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter(mpi_request_, receive_buffer, root, communicator, environment); }

    template <typename Value>
    void scatter(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter_in_place(mpi_request_, send_buffer, root, communicator, environment); }

    template <typename ReceiveValue>
    void scatter(
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter(mpi_request_, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void scatter(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter(mpi_request_, send_buffer, communicator, environment); }

    void scatter(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter(mpi_request_, communicator, environment); }

    // noncontiguous_scatter
    template <typename SendValue, typename ReceiveValue>
    void noncontiguous_scatter(
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_scatter(mpi_request_, send_buffer, receive_buffer, root, communicator, environment); }

    template <typename ReceiveValue>
    void noncontiguous_scatter(
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_scatter(mpi_request_, receive_buffer, root, communicator, environment); }

    template <typename Value>
    void noncontiguous_scatter(
      ::yampi::in_place_t const,
      ::yampi::noncontiguous_buffer<Value, false> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_scatter_in_place(mpi_request_, send_buffer, root, communicator, environment); }

    template <typename ReceiveValue>
    void noncontiguous_scatter(
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_scatter(mpi_request_, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void noncontiguous_scatter(
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_scatter(mpi_request_, send_buffer, communicator, environment); }

    void noncontiguous_scatter(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_scatter(mpi_request_, communicator, environment); }

    // all_gather
    template <typename SendValue, typename ContiguousIterator>
    void all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather(mpi_request_, send_buffer, first, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather(mpi_request_, send_buffer, receive_buffer, communicator, environment); }

    template <typename Value>
    void all_gather(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather_in_place(mpi_request_, receive_buffer, communicator, environment); }

    template <typename SendValue, typename ContiguousIterator>
    void all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather(mpi_request_, send_buffer, first, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather(mpi_request_, send_buffer, receive_buffer, communicator, environment); }

    // neighbor all_gather
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    void all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather(mpi_request_, send_buffer, first, topology, environment); }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    void all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather(mpi_request_, send_buffer, receive_buffer, topology, environment); }

    // noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue>
    void noncontiguous_all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_all_gather(mpi_request_, send_buffer, receive_buffer, communicator, environment); }

    template <typename Value>
    void noncontiguous_all_gather(
      ::yampi::in_place_t const,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_all_gather_in_place(mpi_request_, receive_buffer, communicator, environment); }

    // neighbor noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue, typename Topology>
    void noncontiguous_all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_all_gather(mpi_request_, send_buffer, receive_buffer, topology, environment); }

    // complete_exchange
    template <typename SendValue, typename ContiguousIterator>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange(mpi_request_, send_buffer, first, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange(mpi_request_, send_buffer, receive_buffer, communicator, environment); }

    template <typename Value>
    void complete_exchange(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange_in_place(mpi_request_, receive_buffer, communicator, environment); }

    template <typename SendValue, typename ContiguousIterator>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange(mpi_request_, send_buffer, first, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange(mpi_request_, send_buffer, receive_buffer, communicator, environment); }

    // neighbor complete_exchange
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange(mpi_request_, send_buffer, first, topology, environment); }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange(mpi_request_, send_buffer, receive_buffer, topology, environment); }

    // noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, bool enables_multiple_datatypes>
    void noncontiguous_complete_exchange(
      ::yampi::noncontiguous_buffer<SendValue, enables_multiple_datatypes> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, enables_multiple_datatypes> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_complete_exchange(mpi_request_, send_buffer, receive_buffer, communicator, environment); }

    template <typename Value, bool enables_multiple_datatypes>
    void noncontiguous_complete_exchange(
      ::yampi::in_place_t const,
      ::yampi::noncontiguous_buffer<Value, enables_multiple_datatypes> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_complete_exchange_in_place(mpi_request_, receive_buffer, communicator, environment); }

    // neighbor noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, bool enables_multiple_datatypes, typename Topology>
    void noncontiguous_complete_exchange(
      ::yampi::noncontiguous_buffer<SendValue, enables_multiple_datatypes> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, enables_multiple_datatypes> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_complete_exchange(mpi_request_, send_buffer, receive_buffer, topology, environment); }

    // reduce
    template <typename SendValue, typename ContiguousIterator>
    void reduce(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce(mpi_request_, send_buffer, first, operation, root, communicator, environment); }

    template <typename SendValue>
    void reduce(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce(mpi_request_, send_buffer, operation, root, communicator, environment); }

    template <typename Value>
    void reduce(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce_in_place(mpi_request_, buffer, operation, root, communicator, environment); }

    template <typename SendValue>
    void reduce(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce(mpi_request_, send_buffer, operation, root, communicator, environment); }

    template <typename ReceiveValue>
    void reduce(
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce(mpi_request_, receive_buffer, operation, communicator, environment); }

    void reduce(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce(mpi_request_, communicator, environment); }

    // all_reduce
    template <typename SendValue, typename ContiguousIterator>
    void all_reduce(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_reduce(mpi_request_, send_buffer, first, operation, communicator, environment); }

    template <typename Value>
    void all_reduce(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_reduce_in_place(mpi_request_, buffer, operation, communicator, environment); }

    // reduce_scatter
    template <typename SendValue, typename ContiguousIterator>
    void reduce_scatter(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce_scatter(mpi_request_, send_buffer, first, operation, communicator, environment); }

    template <typename Value>
    void reduce_scatter(
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce_scatter(mpi_request_, send_buffer, receive_buffer, operation, communicator, environment); }

    template <typename Value>
    void reduce_scatter(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce_scatter_in_place(mpi_request_, buffer, operation, communicator, environment); }

    // inclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    void inclusive_scan(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::inclusive_scan(mpi_request_, send_buffer, first, operation, communicator, environment); }

    template <typename Value>
    void inclusive_scan(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::inclusive_scan_in_place(mpi_request_, buffer, operation, communicator, environment); }

    // exclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    void exclusive_scan(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::exclusive_scan(mpi_request_, send_buffer, first, operation, communicator, environment); }

    template <typename Value>
    void exclusive_scan(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::exclusive_scan_in_place(mpi_request_, buffer, operation, communicator, environment); }

    // duplicate communicator
    void duplicate_communicator(
      ::yampi::communicator_base const& old_communicator, ::yampi::communicator_base& new_communicator,
      ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::duplicate_communicator(mpi_request_, old_communicator, new_communicator, environment); }
#   if MPI_VERSION >= 4

    void duplicate_communicator(
      ::yampi::communicator_base const& old_communicator, ::yampi::information const& information,
      ::yampi::communicator_base& new_communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::duplicate_communicator(mpi_request_, old_communicator, information, new_communicator, environment); }
#   endif // MPI_VERSION >= 4
# endif // MPI_VERSION >= 3
  };

  inline void swap(::yampi::immediate_request& lhs, ::yampi::immediate_request& rhs) noexcept
  { lhs.swap(rhs); }

  class immediate_request_ref
    : public ::yampi::request_ref_base
  {
    typedef ::yampi::request_ref_base base_type;

   public:
    immediate_request_ref() = delete;
    ~immediate_request_ref() noexcept = default;

    using base_type::base_type;
    using base_type::reset;

    void reset(::yampi::immediate_request&& request, ::yampi::environment const& environment)
    {
      free(environment);
      *mpi_request_ptr_ = std::move(request.mpi_request_);
      request.mpi_request_ = MPI_REQUEST_NULL;
    }

    // send
    template <typename Value>
    void reset(
      ::yampi::request_send_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send(buffer, destination, tag, communicator, environment);
    }

    template <typename Mode, typename Value>
    void reset(
      ::yampi::request_send_t const, Mode const mode,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send(mode, buffer, destination, tag, communicator, environment);
    }

    // receive
    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, communicator, environment);
    }
# if MPI_VERSION >= 3

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::message& message,
      ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, message, environment);
    }
# endif // MPI_VERSION >= 3

# if MPI_VERSION >= 4
    // send_receive
    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send_receive(send_buffer, destination, send_tag, receive_buffer, source, receive_tag, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send_receive(send_buffer, destination, send_tag, receive_buffer, source, ::yampi::any_tag, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send_receive(send_buffer, destination, send_tag, receive_buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment);
    }

    // with replacement
    template <typename Value>
    void reset(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send_receive(buffer, destination, send_tag, source, receive_tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send_receive(buffer, destination, send_tag, source, ::yampi::any_tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_send_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send_receive(buffer, destination, send_tag, ::yampi::any_source, ::yampi::any_tag, communicator, environment);
    }
# endif // MPI_VERSION >= 4

# if MPI_VERSION >= 3
    // Nonblocking collective operations
    // barrier
    void reset(
      ::yampi::request_barrier_t const,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      barrier(communicator, environment);
    }

    // broadcast
    template <typename Value>
    void reset(
      ::yampi::request_broadcast_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const root,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      broadcast(buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_broadcast_t const,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      broadcast(send_buffer, communicator, environment);
    }

    void reset(
      ::yampi::request_broadcast_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      broadcast(communicator, environment);
    }

    // gather
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(send_buffer, first, root, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(send_buffer, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(send_buffer, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_gather_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(in_place, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(send_buffer, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_gather_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(receive_buffer, communicator, environment);
    }

    void reset(
      ::yampi::request_gather_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(communicator, environment);
    }

    // noncontiguous_gather
    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_gather(send_buffer, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_gather(send_buffer, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_gather(in_place, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_gather(send_buffer, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_gather(receive_buffer, communicator, environment);
    }

    void reset(
      ::yampi::request_noncontiguous_gather_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_gather(communicator, environment);
    }

    // scatter
    template <typename ContiguousIterator, typename ReceiveValue>
    void reset(
      ::yampi::request_scatter_t const,
      ContiguousIterator const first, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(first, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_scatter_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(send_buffer, receive_buffer, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_scatter_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(receive_buffer, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_scatter_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(in_place, send_buffer, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_scatter_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_scatter_t const,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(send_buffer, communicator, environment);
    }

    void reset(
      ::yampi::request_scatter_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      scatter(communicator, environment);
    }

    // noncontiguous_scatter
    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_scatter(send_buffer, receive_buffer, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_scatter(receive_buffer, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::noncontiguous_buffer<Value, false> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_scatter(in_place, send_buffer, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_scatter(receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_scatter(send_buffer, communicator, environment);
    }

    void reset(
      ::yampi::request_noncontiguous_scatter_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_scatter(communicator, environment);
    }

    // all_gather
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(send_buffer, first, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(send_buffer, receive_buffer, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(in_place, receive_buffer, communicator, environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(send_buffer, first, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(send_buffer, receive_buffer, communicator, environment);
    }

    // neighbor all_gather
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(send_buffer, first, topology, environment);
    }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    void reset(
      ::yampi::request_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      all_gather(send_buffer, receive_buffer, topology, environment);
    }

    // noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_noncontiguous_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_all_gather(send_buffer, receive_buffer, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_noncontiguous_all_gather_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_all_gather(in_place, receive_buffer, communicator, environment);
    }

    // neighbor noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue, typename Topology>
    void reset(
      ::yampi::request_noncontiguous_all_gather_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_all_gather(send_buffer, receive_buffer, topology, environment);
    }

    // complete_exchange
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, first, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, receive_buffer, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(in_place, receive_buffer, communicator, environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, first, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, receive_buffer, communicator, environment);
    }

    // neighbor complete_exchange
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, first, topology, environment);
    }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    void reset(
      ::yampi::request_complete_exchange_t const,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, receive_buffer, topology, environment);
    }

    // noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, bool enables_multiple_datatypes>
    void reset(
      ::yampi::request_noncontiguous_complete_exchange_t const,
      ::yampi::noncontiguous_buffer<SendValue, enables_multiple_datatypes> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, enables_multiple_datatypes> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_complete_exchange(send_buffer, receive_buffer, communicator, environment);
    }

    template <typename Value, bool enables_multiple_datatypes>
    void reset(
      ::yampi::request_noncontiguous_complete_exchange_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::noncontiguous_buffer<Value, enables_multiple_datatypes> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_complete_exchange(in_place, receive_buffer, communicator, environment);
    }

    // neighbor noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, bool enables_multiple_datatypes, typename Topology>
    void reset(
      ::yampi::request_noncontiguous_complete_exchange_t const,
      ::yampi::noncontiguous_buffer<SendValue, enables_multiple_datatypes> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, enables_multiple_datatypes> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      noncontiguous_complete_exchange(send_buffer, receive_buffer, topology, environment);
    }

    // reduce
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_reduce_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(send_buffer, first, operation, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_reduce_t const,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(send_buffer, operation, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_reduce_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(in_place, buffer, operation, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::request_reduce_t const,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(send_buffer, operation, root, communicator, environment);
    }

    template <typename ReceiveValue>
    void reset(
      ::yampi::request_reduce_t const,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(receive_buffer, operation, communicator, environment);
    }

    void reset(
      ::yampi::request_reduce_t const,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(communicator, environment);
    }

    // all_reduce
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_all_reduce_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_reduce(send_buffer, first, operation, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_all_reduce_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_reduce(in_place, buffer, operation, communicator, environment);
    }

    // reduce_scatter
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_reduce_scatter_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce_scatter(send_buffer, first, operation, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_reduce_scatter_t const,
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce_scatter(send_buffer, receive_buffer, operation, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_reduce_scatter_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce_scatter(in_place, buffer, operation, communicator, environment);
    }

    // inclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_inclusive_scan_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      inclusive_scan(send_buffer, first, operation, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_inclusive_scan_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      inclusive_scan(in_place, buffer, operation, communicator, environment);
    }

    // exclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::request_exclusive_scan_t const,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      exclusive_scan(send_buffer, first, operation, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_exclusive_scan_t const,
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      exclusive_scan(in_place, buffer, operation, communicator, environment);
    }

    // duplicate communicator
    template <typename Value>
    void reset(
      ::yampi::request_duplicate_communicator_t const,
      ::yampi::communicator_base const& old_communicator, ::yampi::communicator_base& new_communicator,
      ::yampi::environment const& environment)
    {
      free(environment);
      duplicate_communicator(old_communicator, new_communicator, environment);
    }
#   if MPI_VERSION >= 4

    template <typename Value>
    void reset(
      ::yampi::request_duplicate_communicator_t const,
      ::yampi::communicator_base const& old_communicator, ::yampi::information const& information,
      ::yampi::communicator_base& new_communicator, ::yampi::environment const& environment)
    {
      free(environment);
      duplicate_communicator(old_communicator, information, new_communicator, environment);
    }
#   endif // MPI_VERSION >= 4
# endif // MPI_VERSION >= 3

    // send
    template <typename Value>
    void send(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::standard_send(*mpi_request_ptr_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::standard_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::standard_send(*mpi_request_ptr_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::buffered_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::buffered_send(*mpi_request_ptr_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::synchronous_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::synchronous_send(*mpi_request_ptr_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::ready_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::ready_send(*mpi_request_ptr_, buffer, destination, tag, communicator, environment); }

    // receive
    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::receive(*mpi_request_ptr_, buffer, source, tag, communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::receive(*mpi_request_ptr_, buffer, source, ::yampi::any_tag, communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::receive(*mpi_request_ptr_, buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment); }
# if MPI_VERSION >= 3

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer, ::yampi::message& message,
      ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::receive(*mpi_request_ptr_, buffer, message, environment); }
# endif // MPI_VERSION >= 3

# if MPI_VERSION >= 4
    // send_receive
    template <typename SendValue, typename ReceiveValue>
    void send_receive(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::send_receive(*mpi_request_ptr_, send_buffer, destination, send_tag, receive_buffer, source, receive_tag, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void send_receive(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::send_receive(*mpi_request_ptr_, send_buffer, destination, send_tag, receive_buffer, source, ::yampi::any_tag, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void send_receive(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::send_receive(*mpi_request_ptr_, send_buffer, destination, send_tag, receive_buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment); }

    // with replacement
    template <typename Value>
    void send_receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::send_receive(*mpi_request_ptr_, buffer, destination, send_tag, source, receive_tag, communicator, environment); }

    template <typename Value>
    void send_receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::send_receive(*mpi_request_ptr_, buffer, destination, send_tag, source, ::yampi::any_tag, communicator, environment); }

    template <typename Value>
    void send_receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::send_receive(*mpi_request_ptr_, buffer, destination, send_tag, ::yampi::any_source, ::yampi::any_tag, communicator, environment); }
# endif // MPI_VERSION >= 4

# if MPI_VERSION >= 3
    // Nonblocking collective operations
    // barrier
    void barrier(::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::barrier(*mpi_request_ptr_, communicator, environment); }

    // broadcast
    template <typename Value>
    void broadcast(
      ::yampi::buffer<Value> buffer, ::yampi::rank const root,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::broadcast(*mpi_request_ptr_, buffer, root, communicator, environment); }

    template <typename SendValue>
    void broadcast(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::broadcast(*mpi_request_ptr_, send_buffer, communicator, environment); }

    void broadcast(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::broadcast(*mpi_request_ptr_, communicator, environment); }

    // gather
    template <typename SendValue, typename ContiguousIterator>
    void gather(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather(*mpi_request_ptr_, send_buffer, first, root, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather(*mpi_request_ptr_, send_buffer, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather(*mpi_request_ptr_, send_buffer, root, communicator, environment); }

    template <typename Value>
    void gather(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather_in_place(*mpi_request_ptr_, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather(*mpi_request_ptr_, send_buffer, root, communicator, environment); }

    template <typename ReceiveValue>
    void gather(
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather(*mpi_request_ptr_, receive_buffer, communicator, environment); }

    void gather(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::gather(*mpi_request_ptr_, communicator, environment); }

    // noncontiguous_gather
    template <typename SendValue, typename ReceiveValue>
    void noncontiguous_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_gather(*mpi_request_ptr_, send_buffer, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void noncontiguous_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_gather(*mpi_request_ptr_, send_buffer, root, communicator, environment); }

    template <typename Value>
    void noncontiguous_gather(
      ::yampi::in_place_t const,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_gather_in_place(*mpi_request_ptr_, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void noncontiguous_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_gather(*mpi_request_ptr_, send_buffer, root, communicator, environment); }

    template <typename ReceiveValue>
    void noncontiguous_gather(
      ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_gather(*mpi_request_ptr_, receive_buffer, communicator, environment); }

    void noncontiguous_gather(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_gather(*mpi_request_ptr_, communicator, environment); }

    // scatter
    template <typename ContiguousIterator, typename ReceiveValue>
    void scatter(
      ContiguousIterator const first, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter(*mpi_request_ptr_, first, receive_buffer, root, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void scatter(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter(*mpi_request_ptr_, send_buffer, receive_buffer, root, communicator, environment); }

    template <typename ReceiveValue>
    void scatter(
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter(*mpi_request_ptr_, receive_buffer, root, communicator, environment); }

    template <typename Value>
    void scatter(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter_in_place(*mpi_request_ptr_, send_buffer, root, communicator, environment); }

    template <typename ReceiveValue>
    void scatter(
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter(*mpi_request_ptr_, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void scatter(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter(*mpi_request_ptr_, send_buffer, communicator, environment); }

    void scatter(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::scatter(*mpi_request_ptr_, communicator, environment); }

    // noncontiguous_scatter
    template <typename SendValue, typename ReceiveValue>
    void noncontiguous_scatter(
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_scatter(*mpi_request_ptr_, send_buffer, receive_buffer, root, communicator, environment); }

    template <typename ReceiveValue>
    void noncontiguous_scatter(
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_scatter(*mpi_request_ptr_, receive_buffer, root, communicator, environment); }

    template <typename Value>
    void noncontiguous_scatter(
      ::yampi::in_place_t const,
      ::yampi::noncontiguous_buffer<Value, false> const send_buffer, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_scatter_in_place(*mpi_request_ptr_, send_buffer, root, communicator, environment); }

    template <typename ReceiveValue>
    void noncontiguous_scatter(
      ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_scatter(*mpi_request_ptr_, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void noncontiguous_scatter(
      ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_scatter(*mpi_request_ptr_, send_buffer, communicator, environment); }

    void noncontiguous_scatter(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_scatter(*mpi_request_ptr_, communicator, environment); }

    // all_gather
    template <typename SendValue, typename ContiguousIterator>
    void all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather(*mpi_request_ptr_, send_buffer, first, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather(*mpi_request_ptr_, send_buffer, receive_buffer, communicator, environment); }

    template <typename Value>
    void all_gather(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather_in_place(*mpi_request_ptr_, receive_buffer, communicator, environment); }

    template <typename SendValue, typename ContiguousIterator>
    void all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather(*mpi_request_ptr_, send_buffer, first, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather(*mpi_request_ptr_, send_buffer, receive_buffer, communicator, environment); }

    // neighbor all_gather
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    void all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather(*mpi_request_ptr_, send_buffer, first, topology, environment); }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    void all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_gather(*mpi_request_ptr_, send_buffer, receive_buffer, topology, environment); }

    // noncontiguous_all_gather
    template <typename SendValue, typename ReceiveValue>
    void noncontiguous_all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_all_gather(*mpi_request_ptr_, send_buffer, receive_buffer, communicator, environment); }

    template <typename Value>
    void noncontiguous_all_gather(
      ::yampi::in_place_t const,
      ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_all_gather_in_place(*mpi_request_ptr_, receive_buffer, communicator, environment); }

    // neighbor all_gather
    template <typename SendValue, typename ReceiveValue, typename Topology>
    void noncontiguous_all_gather(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_all_gather(*mpi_request_ptr_, send_buffer, receive_buffer, topology, environment); }

    // complete_exchange
    template <typename SendValue, typename ContiguousIterator>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange(*mpi_request_ptr_, send_buffer, first, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange(*mpi_request_ptr_, send_buffer, receive_buffer, communicator, environment); }

    template <typename Value>
    void complete_exchange(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange_in_place(*mpi_request_ptr_, receive_buffer, communicator, environment); }

    template <typename SendValue, typename ContiguousIterator>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange(*mpi_request_ptr_, send_buffer, first, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange(*mpi_request_ptr_, send_buffer, receive_buffer, communicator, environment); }

    // neighbor complete_exchange
    template <typename SendValue, typename ContiguousIterator, typename Topology>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange(*mpi_request_ptr_, send_buffer, first, topology, environment); }

    template <typename SendValue, typename ReceiveValue, typename Topology>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::complete_exchange(*mpi_request_ptr_, send_buffer, receive_buffer, topology, environment); }

    // noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, bool enables_multiple_datatypes>
    void noncontiguous_complete_exchange(
      ::yampi::noncontiguous_buffer<SendValue, enables_multiple_datatypes> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, enables_multiple_datatypes> receive_buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_complete_exchange(*mpi_request_ptr_, send_buffer, receive_buffer, communicator, environment); }

    template <typename Value, bool enables_multiple_datatypes>
    void noncontiguous_complete_exchange(
      ::yampi::in_place_t const,
      ::yampi::noncontiguous_buffer<Value, enables_multiple_datatypes> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_complete_exchange_in_place(*mpi_request_ptr_, receive_buffer, communicator, environment); }

    // neighbor noncontiguous_complete_exchange
    template <typename SendValue, typename ReceiveValue, bool enables_multiple_datatypes, typename Topology>
    void noncontiguous_complete_exchange(
      ::yampi::noncontiguous_buffer<SendValue, enables_multiple_datatypes> const send_buffer,
      ::yampi::noncontiguous_buffer<ReceiveValue, enables_multiple_datatypes> receive_buffer,
      ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::noncontiguous_complete_exchange(*mpi_request_ptr_, send_buffer, receive_buffer, topology, environment); }

    // reduce
    template <typename SendValue, typename ContiguousIterator>
    void reduce(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce(*mpi_request_ptr_, send_buffer, first, operation, root, communicator, environment); }

    template <typename SendValue>
    void reduce(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce(*mpi_request_ptr_, send_buffer, operation, root, communicator, environment); }

    template <typename Value>
    void reduce(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce_in_place(*mpi_request_ptr_, buffer, operation, root, communicator, environment); }

    template <typename SendValue>
    void reduce(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const root,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce(*mpi_request_ptr_, send_buffer, operation, root, communicator, environment); }

    template <typename ReceiveValue>
    void reduce(
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce(*mpi_request_ptr_, receive_buffer, operation, communicator, environment); }

    void reduce(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce(*mpi_request_ptr_, communicator, environment); }

    // all_reduce
    template <typename SendValue, typename ContiguousIterator>
    void all_reduce(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_reduce(*mpi_request_ptr_, send_buffer, first, operation, communicator, environment); }

    template <typename Value>
    void all_reduce(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::all_reduce_in_place(*mpi_request_ptr_, buffer, operation, communicator, environment); }

    // reduce_scatter
    template <typename SendValue, typename ContiguousIterator>
    void reduce_scatter(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce_scatter(*mpi_request_ptr_, send_buffer, first, operation, communicator, environment); }

    template <typename Value>
    void reduce_scatter(
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce_scatter(*mpi_request_ptr_, send_buffer, receive_buffer, operation, communicator, environment); }

    template <typename Value>
    void reduce_scatter(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::reduce_scatter_in_place(*mpi_request_ptr_, buffer, operation, communicator, environment); }

    // inclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    void inclusive_scan(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::inclusive_scan(*mpi_request_ptr_, send_buffer, first, operation, communicator, environment); }

    template <typename Value>
    void inclusive_scan(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::inclusive_scan_in_place(*mpi_request_ptr_, buffer, operation, communicator, environment); }

    // exclusive_scan
    template <typename SendValue, typename ContiguousIterator>
    void exclusive_scan(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::exclusive_scan(*mpi_request_ptr_, send_buffer, first, operation, communicator, environment); }

    template <typename Value>
    void exclusive_scan(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::exclusive_scan_in_place(*mpi_request_ptr_, buffer, operation, communicator, environment); }

    // duplicate communicator
    void duplicate_communicator(
      ::yampi::communicator_base const& old_communicator, ::yampi::communicator_base& new_communicator,
      ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::duplicate_communicator(*mpi_request_ptr_, old_communicator, new_communicator, environment); }
#   if MPI_VERSION >= 4

    void duplicate_communicator(
      ::yampi::communicator_base const& old_communicator, ::yampi::information const& information,
      ::yampi::communicator_base& new_communicator, ::yampi::environment const& environment)
    { ::yampi::immediate_request_detail::duplicate_communicator(*mpi_request_ptr_, old_communicator, information, new_communicator, environment); }
#   endif // MPI_VERSION >= 4
# endif // MPI_VERSION >= 3
  };

  inline void swap(::yampi::immediate_request_ref& lhs, ::yampi::immediate_request_ref& rhs) noexcept
  { lhs.swap(rhs); }

  class immediate_request_cref
    : public ::yampi::request_cref_base
  {
    typedef ::yampi::request_cref_base base_type;

   public:
    immediate_request_cref() = delete;
    ~immediate_request_cref() noexcept = default;

    using base_type::base_type;
  };
}


#endif

