#ifndef YAMPI_ALGORITHM_TRANSFORM_HPP
# define YAMPI_ALGORITHM_TRANSFORM_HPP

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
    template <typename Value, typename UnaryFunction, typename ContiguousIterator>
    inline boost::optional< ::yampi::status >
    transform(
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      UnaryFunction unary_function, ContiguousIterator const transform_buffer_first,
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
        std::transform(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          transform_buffer_first, unary_function);

        ::yampi::send(
          ::yampi::make_buffer(
            transform_buffer_first, transform_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }

      return boost::none;
    }

    template <typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      UnaryFunction unary_function,
      ::yampi::message_envelope const message_envelope, ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer(send_buffer.count());
      return ::yampi::algorithm::transform(
        send_buffer, receive_buffer, unary_function, transform_buffer.begin(), message_envelope, environment);
    }

    template <typename CommunicationMode, typename Value, typename UnaryFunction, typename ContiguousIterator>
    inline boost::optional< ::yampi::status >
    transform(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      UnaryFunction unary_function, ContiguousIterator const transform_buffer_first,
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
        std::transform(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          transform_buffer_first, unary_function);

        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          ::yampi::make_buffer(
            transform_buffer_first, transform_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }

      return boost::none;
    }

    template <typename CommunicationMode, typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      UnaryFunction unary_function,
      ::yampi::message_envelope const message_envelope, ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer(send_buffer.count());
      return ::yampi::algorithm::transform(
        YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, unary_function, transform_buffer.begin(), message_envelope, environment);
    }

    // ignoring status
    template <typename Value, typename UnaryFunction, typename ContiguousIterator>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      UnaryFunction unary_function, ContiguousIterator const transform_buffer_first,
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
        std::transform(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          transform_buffer_first, unary_function);

        ::yampi::send(
          ::yampi::make_buffer(
            transform_buffer_first, transform_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }
    }

    template <typename Value, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      UnaryFunction unary_function,
      ::yampi::message_envelope const message_envelope, ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer(send_buffer.count());
      ::yampi::algorithm::transform(
        ignore_status, send_buffer, receive_buffer, unary_function, transform_buffer.begin(), message_envelope, environment);
    }

    template <typename CommunicationMode, typename Value, typename UnaryFunction, typename ContiguousIterator>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      UnaryFunction unary_function, ContiguousIterator const transform_buffer_first,
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
        std::transform(
          send_buffer.data(), send_buffer.data() + send_buffer.count(),
          transform_buffer_first, unary_function);

        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          ::yampi::make_buffer(
            transform_buffer_first, transform_buffer_first + send_buffer.count(),
            send_buffer.datatype()),
          message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
      }
    }

    template <typename CommunicationMode, typename Value, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::buffer<Value> const send_buffer, ::yampi::buffer<Value> receive_buffer,
      UnaryFunction unary_function,
      ::yampi::message_envelope const message_envelope, ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer(send_buffer.count());
      ::yampi::algorithm::transform(
        ignore_status, YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, unary_function,
        transform_buffer.begin(), message_envelope, environment);
    }
  }
}


# undef YAMPI_FORWARD_OR_COPY
# undef YAMPI_RVALUE_REFERENCE_OR_COPY

#endif

