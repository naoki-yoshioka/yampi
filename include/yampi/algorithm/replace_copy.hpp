#ifndef YAMPI_ALGORITHM_REPLACE_COPY_HPP
# define YAMPI_ALGORITHM_REPLACE_COPY_HPP

# include <cassert>
# include <vector>
# include <algorithm>

# include <boost/optional.hpp>
# include <boost/none.hpp>

# include <yampi/send.hpp>
# include <yampi/receive.hpp>
# include <yampi/allocator.hpp>
# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/message_envelope.hpp>
# include <yampi/rank.hpp>
# include <yampi/status.hpp>
# include <yampi/communication_mode.hpp>


namespace yampi
{
  namespace algorithm
  {
    template <typename Value, typename ContiguousIterator>
    inline boost::optional< ::yampi::status >
    replace_copy(
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::buffer<Value> receive_buffer,
      Value const& old_value, Value const& new_value,
      ContiguousIterator const replace_copy_buffer_first,
      ::yampi::message_envelope const message_envelope, ::yampi::environment const& environment)
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (message_envelope.source() == message_envelope.destination())
        return boost::none;

      ::yampi::rank const present_rank = message_envelope.communicator().rank(environment);

      if (present_rank == message_envelope.destination())
        return boost::make_optional(
          ::yampi::receive(
            receive_buffer, message_envelope.source(), message_envelope.tag(),
            message_envelope.communicator(), environment));
      else if (present_rank == message_envelope.source())
      {
        std::replace_copy(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          replace_copy_buffer_first, old_value, new_value);

        ::yampi::send(
          ::yampi::make_buffer(
            replace_copy_buffer_first, replace_copy_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }

      return boost::none;
    }

    template <typename Value>
    inline boost::optional< ::yampi::status >
    replace_copy(
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::buffer<Value> receive_buffer,
      Value const& old_value, Value const& new_value,
      ::yampi::message_envelope const message_envelope, ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > replace_copy_buffer(send_buffer.count());
      return ::yampi::algorithm::replace_copy(
        send_buffer, receive_buffer, old_value, new_value,
        replace_copy_buffer.begin(), message_envelope, environment);
    }

    template <typename CommunicationMode, typename Value, typename ContiguousIterator>
    inline boost::optional< ::yampi::status >
    replace_copy(
      CommunicationMode&& communication_mode,
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::buffer<Value> receive_buffer,
      Value const& old_value, Value const& new_value,
      ContiguousIterator const replace_copy_buffer_first,
      ::yampi::message_envelope const message_envelope, ::yampi::environment const& environment)
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (message_envelope.source() == message_envelope.destination())
        return boost::none;

      ::yampi::rank const present_rank = message_envelope.communicator().rank(environment);

      if (present_rank == message_envelope.destination())
        return boost::make_optional(
          ::yampi::receive(
            receive_buffer, message_envelope.source(), message_envelope.tag(),
            message_envelope.communicator(), environment));
      else if (present_rank == message_envelope.source())
      {
        std::replace_copy(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          replace_copy_buffer_first, old_value, new_value);

        ::yampi::send(
          std::forward<CommunicationMode>(communication_mode),
          ::yampi::make_buffer(
            replace_copy_buffer_first, replace_copy_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }

      return boost::none;
    }

    template <typename CommunicationMode, typename Value>
    inline boost::optional< ::yampi::status >
    replace_copy(
      CommunicationMode&& communication_mode,
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::buffer<Value> receive_buffer,
      Value const& old_value, Value const& new_value,
      ::yampi::message_envelope const message_envelope, ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > replace_copy_buffer(send_buffer.count());
      return ::yampi::algorithm::replace_copy(
        std::forward<CommunicationMode>(communication_mode),
        send_buffer, receive_buffer, old_value, new_value,
        replace_copy_buffer.begin(), message_envelope, environment);
    }

    // ignoring status
    template <typename Value, typename ContiguousIterator>
    inline void replace_copy(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::buffer<Value> receive_buffer,
      Value const& old_value, Value const& new_value,
      ContiguousIterator const replace_copy_buffer_first,
      ::yampi::message_envelope const message_envelope, ::yampi::environment const& environment)
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (message_envelope.source() == message_envelope.destination())
        return;

      ::yampi::rank const present_rank = message_envelope.communicator().rank(environment);

      if (present_rank == message_envelope.destination())
        ::yampi::receive(
          ignore_status,
          receive_buffer, message_envelope.source(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      else if (present_rank == message_envelope.source())
      {
        std::replace_copy(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          replace_copy_buffer_first, old_value, new_value);

        ::yampi::send(
          ::yampi::make_buffer(
            replace_copy_buffer_first, replace_copy_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }
    }

    template <typename Value>
    inline void replace_copy(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::buffer<Value> receive_buffer,
      Value const& old_value, Value const& new_value,
      ::yampi::message_envelope const message_envelope, ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > replace_copy_buffer(send_buffer.count());
      ::yampi::algorithm::replace_copy(
        ignore_status,
        send_buffer, receive_buffer, old_value, new_value,
        replace_copy_buffer.begin(), message_envelope, environment);
    }

    template <typename CommunicationMode, typename Value, typename ContiguousIterator>
    inline void replace_copy(
      ::yampi::ignore_status_t const ignore_status,
      CommunicationMode&& communication_mode,
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::buffer<Value> receive_buffer,
      Value const& old_value, Value const& new_value,
      ContiguousIterator const replace_copy_buffer_first,
      ::yampi::message_envelope const message_envelope, ::yampi::environment const& environment)
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (message_envelope.source() == message_envelope.destination())
        return;

      ::yampi::rank const present_rank = message_envelope.communicator().rank(environment);

      if (present_rank == message_envelope.destination())
        ::yampi::receive(
          ignore_status,
          receive_buffer, message_envelope.source(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      else if (present_rank == message_envelope.source())
      {
        std::replace_copy(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          replace_copy_buffer_first, old_value, new_value);

        ::yampi::send(
          std::forward<CommunicationMode>(communication_mode),
          ::yampi::make_buffer(
            replace_copy_buffer_first, replace_copy_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }
    }

    template <typename CommunicationMode, typename Value>
    inline void replace_copy(
      ::yampi::ignore_status_t const ignore_status,
      CommunicationMode&& communication_mode,
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::buffer<Value> receive_buffer,
      Value const& old_value, Value const& new_value,
      ::yampi::message_envelope const message_envelope, ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > replace_copy_buffer(send_buffer.count());
      ::yampi::algorithm::replace_copy(
        ignore_status, std::forward<CommunicationMode>(communication_mode),
        send_buffer, receive_buffer, old_value, new_value,
        replace_copy_buffer.begin(), message_envelope, environment);
    }
  }
}


#endif

