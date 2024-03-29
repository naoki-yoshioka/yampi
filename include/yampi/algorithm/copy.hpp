#ifndef YAMPI_ALGORITHM_COPY_HPP
# define YAMPI_ALGORITHM_COPY_HPP

# include <cassert>
# include <utility>

# include <boost/optional.hpp>

# include <yampi/send.hpp>
# include <yampi/receive.hpp>
# include <yampi/environment.hpp>
# include <yampi/rank.hpp>
# include <yampi/status.hpp>
# include <yampi/message_envelope.hpp>
# include <yampi/communication_mode.hpp>


namespace yampi
{
  namespace algorithm
  {
    template <typename Value>
    inline boost::optional< ::yampi::status >
    copy(
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::message_envelope const message_envelope,
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
        ::yampi::send(
          send_buffer, message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);

      return boost::none;
    }

    template <typename CommunicationMode, typename Value>
    inline boost::optional< ::yampi::status >
    copy(
      CommunicationMode&& communication_mode,
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::message_envelope const message_envelope,
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
        ::yampi::send(
          std::forward<CommunicationMode>(communication_mode),
          send_buffer, message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);

      return boost::none;
    }

    // ignoring status
    template <typename Value>
    inline void copy(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::message_envelope const message_envelope,
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
        ::yampi::send(
          send_buffer, message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
    }

    template <typename CommunicationMode, typename Value>
    inline void copy(
      ::yampi::ignore_status_t const ignore_status,
      CommunicationMode&& communication_mode,
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::message_envelope const message_envelope,
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
        ::yampi::send(
          std::forward<CommunicationMode>(communication_mode),
          send_buffer, message_envelope.destination(), message_envelope.tag(),
          message_envelope.communicator(), environment);
    }
  }
}


#endif

