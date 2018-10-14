#ifndef YAMPI_ALGORITHM_TRANSFORM_HPP
# define YAMPI_ALGORITHM_TRANSFORM_HPP

# include <boost/config.hpp>

# include <cassert>
# include <vector>
# include <algorithm>

# include <boost/optional.hpp>
# include <boost/none.hpp>

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>

# include <yampi/send.hpp>
# include <yampi/receive.hpp>
# include <yampi/allocator.hpp>
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
    template <typename Value, typename Allocator, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return boost::none;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        return boost::make_optional(
          ::yampi::receive(
            receive_buffer.buffer(), send_buffer.rank(), tag, communicator, environment));
      else if (present_rank == send_buffer.rank())
      {
        transform_buffer.clear();
        transform_buffer.reserve(send_buffer.count());
        std::transform(
          send_buffer.data(), send_buffer.data()+send_buffer.count(),
          std::back_inserter(transform_buffer), unary_function);

        ::yampi::send(
          ::yampi::make_buffer(
            boost::begin(transform_buffer), boost::end(transform_buffer), send_buffer.datatype()),
          receive_buffer.rank(), tag,
          communicator, environment);
      }

      return boost::none;
    }

    template <typename Value, typename Allocator, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      return ::yampi::algorithm::transform(
        send_buffer, receive_buffer, transform_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer;
      return ::yampi::algorithm::transform(
        send_buffer, receive_buffer, transform_buffer, tag, unary_function,
        communicator, environment);
    }

    template <typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      return ::yampi::algorithm::transform(
        send_buffer, receive_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename Value, typename Allocator, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return boost::none;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        return boost::make_optional(
          ::yampi::receive(
            receive_buffer.buffer(), send_buffer.rank(), tag, communicator, environment));
      else if (present_rank == send_buffer.rank())
      {
        transform_buffer.clear();
        transform_buffer.reserve(send_buffer.count());
        std::transform(
          send_buffer.data(), send_buffer.data()+send_buffer.count(),
          std::back_inserter(transform_buffer), unary_function);

        ::yampi::send(
          ::yampi::make_buffer(
            boost::begin(transform_buffer), boost::end(transform_buffer), send_buffer.datatype()),
          receive_buffer.rank(), tag,
          communicator, environment);
      }

      return boost::none;
    }

    template <typename Value, typename Allocator, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      return ::yampi::algorithm::transform(
        send_buffer, receive_buffer, transform_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer;
      return ::yampi::algorithm::transform(
        send_buffer, receive_buffer, transform_buffer, tag, unary_function,
        communicator, environment);
    }

    template <typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      return ::yampi::algorithm::transform(
        send_buffer, receive_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename CommunicationMode, typename Value, typename Allocator, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return boost::none;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        return boost::make_optional(
          ::yampi::receive(
            receive_buffer.buffer(), send_buffer.rank(), tag, communicator, environment));
      else if (present_rank == send_buffer.rank())
      {
        transform_buffer.clear();
        transform_buffer.reserve(send_buffer.count());
        std::transform(
          send_buffer.data(), send_buffer.data()+send_buffer.count(),
          std::back_inserter(transform_buffer), unary_function);

        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          ::yampi::make_buffer(
            boost::begin(transform_buffer), boost::end(transform_buffer), send_buffer.datatype()),
          receive_buffer.rank(), tag,
          communicator, environment);
      }

      return boost::none;
    }

    template <typename CommunicationMode, typename Value, typename Allocator, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      return ::yampi::algorithm::transform(
        YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, transform_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename CommunicationMode, typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer;
      return ::yampi::algorithm::transform(
        YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, transform_buffer, tag, unary_function,
        communicator, environment);
    }

    template <typename CommunicationMode, typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      return ::yampi::algorithm::transform(
        YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename CommunicationMode, typename Value, typename Allocator, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return boost::none;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        return boost::make_optional(
          ::yampi::receive(
            receive_buffer.buffer(), send_buffer.rank(), tag, communicator, environment));
      else if (present_rank == send_buffer.rank())
      {
        transform_buffer.clear();
        transform_buffer.reserve(send_buffer.count());
        std::transform(
          send_buffer.data(), send_buffer.data()+send_buffer.count(),
          std::back_inserter(transform_buffer), unary_function);

        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          ::yampi::make_buffer(
            boost::begin(transform_buffer), boost::end(transform_buffer), send_buffer.datatype()),
          receive_buffer.rank(), tag,
          communicator, environment);
      }

      return boost::none;
    }

    template <typename CommunicationMode, typename Value, typename Allocator, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      return ::yampi::algorithm::transform(
        YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, transform_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename CommunicationMode, typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer;
      return ::yampi::algorithm::transform(
        YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, transform_buffer, tag, unary_function,
        communicator, environment);
    }

    template <typename CommunicationMode, typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status >
    transform(
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      return ::yampi::algorithm::transform(
        YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }


    // ignoring status
    template <typename Value, typename Allocator, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        ::yampi::receive(
          ignore_status,
          receive_buffer.buffer(), send_buffer.rank(), tag,
          communicator, environment);
      else if (present_rank == send_buffer.rank())
      {
        transform_buffer.clear();
        transform_buffer.reserve(send_buffer.count());
        std::transform(
          send_buffer.data(), send_buffer.data()+send_buffer.count(),
          std::back_inserter(transform_buffer), unary_function);

        ::yampi::send(
          ::yampi::make_buffer(
            boost::begin(transform_buffer), boost::end(transform_buffer), send_buffer.datatype()),
          receive_buffer.rank(), tag,
          communicator, environment);
      }
    }

    template <typename Value, typename Allocator, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      ::yampi::algorithm::transform(
        ignore_status, send_buffer, receive_buffer, transform_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename Value, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer;
      ::yampi::algorithm::transform(
        ignore_status,
        send_buffer, receive_buffer, transform_buffer, tag, unary_function,
        communicator, environment);
    }

    template <typename Value, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      ::yampi::algorithm::transform(
        ignore_status, send_buffer, receive_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename Value, typename Allocator, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        ::yampi::receive(
          ignore_status,
          receive_buffer.buffer(), send_buffer.rank(), tag,
          communicator, environment);
      else if (present_rank == send_buffer.rank())
      {
        transform_buffer.clear();
        transform_buffer.reserve(send_buffer.count());
        std::transform(
          send_buffer.data(), send_buffer.data()+send_buffer.count(),
          std::back_inserter(transform_buffer), unary_function);

        ::yampi::send(
          ::yampi::make_buffer(
            boost::begin(transform_buffer), boost::end(transform_buffer), send_buffer.datatype()),
          receive_buffer.rank(), tag,
          communicator, environment);
      }
    }

    template <typename Value, typename Allocator, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      ::yampi::algorithm::transform(
        ignore_status, send_buffer, receive_buffer, transform_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename Value, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer;
      ::yampi::algorithm::transform(
        ignore_status,
        send_buffer, receive_buffer, transform_buffer, tag, unary_function,
        communicator, environment);
    }

    template <typename Value, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      ::yampi::algorithm::transform(
        ignore_status, send_buffer, receive_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename CommunicationMode, typename Value, typename Allocator, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        ::yampi::receive(
          ignore_status,
          receive_buffer.buffer(), send_buffer.rank(), tag,
          communicator, environment);
      else if (present_rank == send_buffer.rank())
      {
        transform_buffer.clear();
        transform_buffer.reserve(send_buffer.count());
        std::transform(
          send_buffer.data(), send_buffer.data()+send_buffer.count(),
          std::back_inserter(transform_buffer), unary_function);

        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          ::yampi::make_buffer(
            boost::begin(transform_buffer), boost::end(transform_buffer), send_buffer.datatype()),
          receive_buffer.rank(), tag,
          communicator, environment);
      }
    }

    template <typename CommunicationMode, typename Value, typename Allocator, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      ::yampi::algorithm::transform(
        ignore_status, YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, transform_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename CommunicationMode, typename Value, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer;
      ::yampi::algorithm::transform(
        ignore_status, YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, transform_buffer, tag, unary_function,
        communicator, environment);
    }

    template <typename CommunicationMode, typename Value, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      ::yampi::algorithm::transform(
        ignore_status, YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename CommunicationMode, typename Value, typename Allocator, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        ::yampi::receive(
          ignore_status,
          receive_buffer.buffer(), send_buffer.rank(), tag,
          communicator, environment);
      else if (present_rank == send_buffer.rank())
      {
        transform_buffer.clear();
        transform_buffer.reserve(send_buffer.count());
        std::transform(
          send_buffer.data(), send_buffer.data()+send_buffer.count(),
          std::back_inserter(transform_buffer), unary_function);

        ::yampi::send(
          YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
          ::yampi::make_buffer(
            boost::begin(transform_buffer), boost::end(transform_buffer), send_buffer.datatype()),
          receive_buffer.rank(), tag,
          communicator, environment);
      }
    }

    template <typename CommunicationMode, typename Value, typename Allocator, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      ::yampi::algorithm::transform(
        ignore_status, YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, transform_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }

    template <typename CommunicationMode, typename Value, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      ::yampi::tag const tag,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer;
      ::yampi::algorithm::transform(
        ignore_status, YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, transform_buffer, tag, unary_function,
        communicator, environment);
    }

    template <typename CommunicationMode, typename Value, typename UnaryFunction>
    inline void transform(
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      UnaryFunction unary_function,
      ::yampi::communicator const& communicator,
      ::yampi::environment const& environment)
    {
      ::yampi::algorithm::transform(
        ignore_status, YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
        send_buffer, receive_buffer, ::yampi::tag(0), unary_function, communicator, environment);
    }
  }
}


# undef YAMPI_FORWARD_OR_COPY
# undef YAMPI_RVALUE_REFERENCE_OR_COPY

#endif

