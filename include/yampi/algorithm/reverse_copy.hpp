#ifndef YAMPI_ALGORITHM_REVERSE_COPY_HPP
# define YAMPI_ALGORITHM_REVERSE_COPY_HPP

# include <boost/config.hpp>

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

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   define YAMPI_RVALUE_REFERENCE_OR_COPY(T) T&&
#   define YAMPI_FORWARD_OR_COPY(T, x) std::forward<T>(x)
# else
#   define YAMPI_RVALUE_REFERENCE_OR_COPY(T) T
#   define YAMPI_FORWARD_OR_COPY(T, x) x
# endif


namespace yampi
{
  namespace algorithm
  {
    template <typename Value, typename ContiguousIterator>
    inline boost::optional< ::yampi::status >
    reverse_copy(
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value>& receive_buffer,
      ContiguousIterator const reverse_copy_buffer_first,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
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
        std::reverse_copy(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          reverse_copy_buffer_first);

        ::yampi::send(
          ::yampi::make_buffer(
            reverse_copy_buffer_first, reverse_copy_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }

      return boost::none;
    }

    template <typename Value>
    inline boost::optional< ::yampi::status >
    reverse_copy(
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value>& receive_buffer,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > reverse_copy_buffer(send_buffer.count());
      return ::yampi::algorithm::reverse_copy(
        send_buffer, receive_buffer,
        reverse_copy_buffer.begin(), message_envelope, environment);
    }

    template <typename Value, typename ContiguousIterator>
    inline boost::optional< ::yampi::status >
    reverse_copy(
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value> const& receive_buffer,
      ContiguousIterator const reverse_copy_buffer_first,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
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
        std::reverse_copy(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          reverse_copy_buffer_first);

        ::yampi::send(
          ::yampi::make_buffer(
            reverse_copy_buffer_first, reverse_copy_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }

      return boost::none;
    }

    template <typename Value>
    inline boost::optional< ::yampi::status >
    reverse_copy(
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value> const& receive_buffer,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > reverse_copy_buffer(send_buffer.count());
      return ::yampi::algorithm::reverse_copy(
        send_buffer, receive_buffer,
        reverse_copy_buffer.begin(), message_envelope, environment);
    }

    template <typename CommunicationMode, typename Value, typename ContiguousIterator>
    inline boost::optional< ::yampi::status >
    reverse_copy(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value>& receive_buffer,
      ContiguousIterator const reverse_copy_buffer_first,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
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
        std::reverse_copy(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          reverse_copy_buffer_first);

        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          ::yampi::make_buffer(
            reverse_copy_buffer_first, reverse_copy_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }

      return boost::none;
    }

    template <typename CommunicationMode, typename Value>
    inline boost::optional< ::yampi::status >
    reverse_copy(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value>& receive_buffer,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > reverse_copy_buffer(send_buffer.count());
      return ::yampi::algorithm::reverse_copy(
        YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer,
        reverse_copy_buffer.begin(), message_envelope, environment);
    }

    template <typename CommunicationMode, typename Value, typename ContiguousIterator>
    inline boost::optional< ::yampi::status >
    reverse_copy(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value> const& receive_buffer,
      ContiguousIterator const reverse_copy_buffer_first,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
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
        std::reverse_copy(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          reverse_copy_buffer_first);

        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          ::yampi::make_buffer(
            reverse_copy_buffer_first, reverse_copy_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }

      return boost::none;
    }

    template <typename CommunicationMode, typename Value>
    inline boost::optional< ::yampi::status >
    reverse_copy(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value> const& receive_buffer,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > reverse_copy_buffer(send_buffer.count());
      return ::yampi::algorithm::reverse_copy(
        YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer,
        reverse_copy_buffer.begin(), message_envelope, environment);
    }


    // ignoring status
    template <typename Value, typename ContiguousIterator>
    inline void reverse_copy(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value>& receive_buffer,
      ContiguousIterator const reverse_copy_buffer_first,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
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
        std::reverse_copy(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          reverse_copy_buffer_first);

        ::yampi::send(
          ::yampi::make_buffer(
            reverse_copy_buffer_first, reverse_copy_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }
    }

    template <typename Value>
    inline void reverse_copy(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value>& receive_buffer,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > reverse_copy_buffer(send_buffer.count());
      ::yampi::algorithm::reverse_copy(
        ignore_status,
        send_buffer, receive_buffer,
        reverse_copy_buffer.begin(), message_envelope, environment);
    }

    template <typename Value, typename ContiguousIterator>
    inline void reverse_copy(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value> const& receive_buffer,
      ContiguousIterator const reverse_copy_buffer_first,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
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
        std::reverse_copy(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          reverse_copy_buffer_first);

        ::yampi::send(
          ::yampi::make_buffer(
            reverse_copy_buffer_first, reverse_copy_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }
    }

    template <typename Value>
    inline void reverse_copy(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value> const& receive_buffer,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > reverse_copy_buffer(send_buffer.count());
      ::yampi::algorithm::reverse_copy(
        ignore_status,
        send_buffer, receive_buffer,
        reverse_copy_buffer.begin(), message_envelope, environment);
    }

    template <typename CommunicationMode, typename Value, typename ContiguousIterator>
    inline void reverse_copy(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value>& receive_buffer,
      ContiguousIterator const reverse_copy_buffer_first,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
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
        std::reverse_copy(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          reverse_copy_buffer_first);

        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          ::yampi::make_buffer(
            reverse_copy_buffer_first, reverse_copy_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }
    }

    template <typename CommunicationMode, typename Value>
    inline void reverse_copy(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value>& receive_buffer,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > reverse_copy_buffer(send_buffer.count());
      ::yampi::algorithm::reverse_copy(
        ignore_status, YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer,
        reverse_copy_buffer.begin(), message_envelope, environment);
    }

    template <typename CommunicationMode, typename Value, typename ContiguousIterator>
    inline void reverse_copy(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value> const& receive_buffer,
      ContiguousIterator const reverse_copy_buffer_first,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
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
        std::reverse_copy(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          reverse_copy_buffer_first);

        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          ::yampi::make_buffer(
            reverse_copy_buffer_first, reverse_copy_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }
    }

    template <typename CommunicationMode, typename Value>
    inline void reverse_copy(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::buffer<Value> const& send_buffer,
      ::yampi::buffer<Value> const& receive_buffer,
      ::yampi::message_envelope const& message_envelope,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > reverse_copy_buffer(send_buffer.count());
      ::yampi::algorithm::reverse_copy(
        ignore_status, YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer,
        reverse_copy_buffer.begin(), message_envelope, environment);
    }
  }
}


# undef YAMPI_FORWARD_OR_COPY
# undef YAMPI_RVALUE_REFERENCE_OR_COPY

#endif

