#ifndef YAMPI_ALGORITHM_COPY_HPP
# define YAMPI_ALGORITHM_COPY_HPP

# include <boost/config.hpp>

# include <cassert>
# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   include <utility>
# endif

# include <boost/optional.hpp>

# include <yampi/send.hpp>
# include <yampi/receive.hpp>
# include <yampi/environment.hpp>
# include <yampi/algorithm/ranked_buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
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
    template <typename Value>
    inline boost::optional< ::yampi::status >
    copy(
      ::yampi::communicator const& communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return boost::none;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        return boost::make_optional(
          ::yampi::receive(
            communicator, environment, receive_buffer.buffer(), send_buffer.rank(), tag));
      else if (present_rank == send_buffer.rank())
        ::yampi::send(
          communicator, environment, send_buffer.buffer(), receive_buffer.rank(), tag);

      return boost::none;
    }

    template <typename Value>
    inline boost::optional< ::yampi::status >
    copy(
      ::yampi::communicator const& communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return boost::none;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        return boost::make_optional(
          ::yampi::receive(
            communicator, environment, receive_buffer.buffer(), send_buffer.rank(), tag));
      else if (present_rank == send_buffer.rank())
        ::yampi::send(
          communicator, environment, send_buffer.buffer(), receive_buffer.rank(), tag);

      return boost::none;
    }

    template <typename CommunicationMode, typename Value>
    inline boost::optional< ::yampi::status >
    copy(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return boost::none;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        return boost::make_optional(
          ::yampi::receive(
            communicator, environment, receive_buffer.buffer(), send_buffer.rank(), tag));
      else if (present_rank == send_buffer.rank())
        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          communicator, environment, send_buffer.buffer(), receive_buffer.rank(), tag);

      return boost::none;
    }

    template <typename CommunicationMode, typename Value>
    inline boost::optional< ::yampi::status >
    copy(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return boost::none;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        return boost::make_optional(
          ::yampi::receive(
            communicator, environment, receive_buffer.buffer(), send_buffer.rank(), tag));
      else if (present_rank == send_buffer.rank())
        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          communicator, environment, send_buffer.buffer(), receive_buffer.rank(), tag);

      return boost::none;
    }


    // ignoring status
    template <typename Value>
    inline void copy(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        ::yampi::receive(
          ignore_status, communicator, environment,
          receive_buffer.buffer(), send_buffer.rank(), tag);
      else if (present_rank == send_buffer.rank())
        ::yampi::send(
          communicator, environment, send_buffer.buffer(), receive_buffer.rank(), tag);
    }

    template <typename Value>
    inline void copy(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        ::yampi::receive(
          ignore_status, communicator, environment,
          receive_buffer.buffer(), send_buffer.rank(), tag);
      else if (present_rank == send_buffer.rank())
        ::yampi::send(
          communicator, environment, send_buffer.buffer(), receive_buffer.rank(), tag);
    }

    template <typename CommunicationMode, typename Value>
    inline void copy(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        ::yampi::receive(
          ignore_status, communicator, environment,
          receive_buffer.buffer(), send_buffer.rank(), tag);
      else if (present_rank == send_buffer.rank())
        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          communicator, environment, send_buffer.buffer(), receive_buffer.rank(), tag);
    }

    template <typename CommunicationMode, typename Value>
    inline void copy(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        ::yampi::receive(
          ignore_status, communicator, environment,
          receive_buffer.buffer(), send_buffer.rank(), tag);
      else if (present_rank == send_buffer.rank())
        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          communicator, environment, send_buffer.buffer(), receive_buffer.rank(), tag);
    }
  }
}


# undef YAMPI_FORWARD_OR_COPY
# undef YAMPI_RVALUE_REFERENCE_OR_COPY

#endif

