#ifndef YAMPI_ALGORITHM_TRANSFORM_HPP
# define YAMPI_ALGORITHM_TRANSFORM_HPP

# include <boost/config.hpp>

# include <cassert>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
#   include <boost/type_traits/is_same.hpp>
# endif
# include <iterator>
# include <vector>
# include <algorithm>

# include <boost/optional.hpp>

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/value_type.hpp>

# include <yampi/blocking_send.hpp>
# include <yampi/blocking_receive.hpp>
# include <yampi/has_corresponding_datatype.hpp>
# include <yampi/is_contiguous_iterator.hpp>
# include <yampi/is_contiguous_range.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/status.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
#   define YAMPI_is_same std::is_same
# else
#   define YAMPI_enable_if boost::enable_if_c
#   define YAMPI_is_same boost::is_same
# endif

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   define YAMPI_RVALUE_REFERENCE_OR_COPY(Type) Type&&
#   define YAMPI_FORWARD_OR_COPY(Type, value) std::forward<Type>(value)
# else
#   define YAMPI_RVALUE_REFERENCE_OR_COPY(Type) Type
#   define YAMPI_FORWARD_OR_COPY(Type, value) value
# endif


namespace yampi
{
  namespace algorithm
  {
    namespace transform_detail
    {
      template <typename Value, typename UnaryFunction>
      inline
      typename YAMPI_enable_if<
        ::yampi::has_corresponding_datatype<Value>::value,
        boost::optional< ::yampi::status> >::type
      transform(
        Value const& send_value, ::yampi::rank const source,
        Value& receive_value, ::yampi::rank const destination,
        ::yampi::tag const tag, ::yampi::communicator const communicator,
        YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
      {
        if (source == destination)
          return boost::none;

        ::yampi::rank present_rank = communicator.rank();

        if (present_rank == destination)
          return boost::make_optional(
            ::yampi::blocking_receive_detail::blocking_receive(
              receive_value, source, tag, communicator));
        else if (present_rank == source)
          ::yampi::blocking_send_detail::blocking_send(
            unary_function(send_value), destination, tag, communicator);

        return boost::none;
      }

      template <typename ContiguousIterator1, typename ContiguousIterator2, typename UnaryFunction>
      inline
      typename YAMPI_enable_if<
        ::yampi::has_corresponding_datatype<
          typename std::iterator_traits<ContiguousIterator1>::value_type>::value
          and YAMPI_is_same<
                typename std::iterator_traits<ContiguousIterator1>::type,
                typename std::iterator_traits<ContiguousIterator2>::type>::value,
        boost::optional< ::yampi::status> >::type
      transform(
        ContiguousIterator1 const send_first, ContiguousIterator1 const send_last,
        ::yampi::rank const source,
        ContiguousIterator2 const receive_first, ::yampi::rank const destination,
        ::yampi::tag const tag, ::yampi::communicator const communicator,
        YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
      {
        assert(send_last >= send_first);

        if (source == destination)
          return boost::none;

        ::yampi::rank present_rank = communicator.rank();

        if (present_rank == destination)
          return boost::make_optional(
            ::yampi::blocking_receive_detail::blocking_receive(
              receive_first, send_last-send_first, source, tag, communicator));
        else if (present_rank == source)
        {
          typedef typename std::iterator_traits<ContiguousIterator1>::value_type value_type;

          std::vector<value_type> buffer;
          buffer.reserve(send_last-send_first);
          std::transform(send_first, send_last, std::back_inserter(buffer), unary_function);

          ::yampi::blocking_send_detail::blocking_send(
            boost::begin(buffer), boost::end(buffer), destination, tag, communicator);
        }

        return boost::none;
      }


      // ignoring status
      template <typename Value, typename UnaryFunction>
      inline
      typename YAMPI_enable_if< ::yampi::has_corresponding_datatype<Value>::value, void>::type
      transform(
        Value const& send_value, ::yampi::rank const source,
        Value& receive_value, ::yampi::rank const destination,
        ::yampi::tag const tag, ::yampi::communicator const communicator,
        ::yampi::ignore_status_t const ignore_status,
        YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
      {
        if (source == destination)
          return;

        ::yampi::rank present_rank = communicator.rank();

        if (present_rank == destination)
          ::yampi::blocking_receive_detail::blocking_receive(
            receive_value, source, tag, communicator, ignore_status);
        else if (present_rank == source)
          ::yampi::blocking_send_detail::blocking_send(
            unary_function(send_value), destination, tag, communicator);
      }

      template <typename ContiguousIterator1, typename ContiguousIterator2, typename UnaryFunction>
      inline
      typename YAMPI_enable_if<
        ::yampi::has_corresponding_datatype<
          typename std::iterator_traits<ContiguousIterator1>::value_type>::value
          and YAMPI_is_same<
                typename std::iterator_traits<ContiguousIterator1>::type,
                typename std::iterator_traits<ContiguousIterator2>::type>::value,
        void>::type
      transform(
        ContiguousIterator1 const send_first, ContiguousIterator1 const send_last,
        ::yampi::rank const source,
        ContiguousIterator2 const receive_first, ::yampi::rank const destination,
        ::yampi::tag const tag, ::yampi::communicator const communicator,
        ::yampi::ignore_status_t const ignore_status,
        YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
      {
        assert(send_last >= send_first);

        if (source == destination)
          return;

        ::yampi::rank present_rank = communicator.rank();

        if (present_rank == destination)
          ::yampi::blocking_receive_detail::blocking_receive(
            receive_first, send_last-send_first, source, tag, communicator, ignore_status);
        else if (present_rank == source)
        {
          typedef typename std::iterator_traits<ContiguousIterator1>::value_type value_type;

          std::vector<value_type> buffer;
          buffer.reserve(send_last-send_first);
          std::transform(send_first, send_last, std::back_inserter(buffer), unary_function);

          ::yampi::blocking_send_detail::blocking_send(
            boost::begin(buffer), boost::end(buffer), destination, tag, communicator);
        }
      }
    } // namespace transform_detail


    template <typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status>
    transform(
      Value const& send_value, ::yampi::rank const source,
      Value& receive_value, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      return ::yampi::algorithm::transform_detail::transform(
        send_value, source, receive_value, destination, tag, communicator,
        YAMPI_FORWARD_OR_COPY(UnaryFunction, unary_function));
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2, typename UnaryFunction>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator1>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator2>::value,
      boost::optional< ::yampi::status> >::type
    transform(
      ContiguousIterator1 const send_first, ContiguousIterator1 const send_last,
      ::yampi::rank const source,
      ContiguousIterator2 const receive_first, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      return ::yampi::algorithm::transform_detail::transform(
        send_first, send_last, source, receive_first, destination,
        tag, communicator, YAMPI_FORWARD_OR_COPY(UnaryFunction, unary_function));
    }

    template <typename ContiguousRange, typename ContiguousIterator, typename UnaryFunction>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      boost::optional< ::yampi::status> >::type
    transform(
      ContiguousRange const& values, ::yampi::rank const source,
      ContiguousIterator const receive_first, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      return ::yampi::algorithm::transform_detail::transform(
        boost::begin(values), boost::end(values), source, receive_first, destination,
        tag, communicator, YAMPI_FORWARD_OR_COPY(UnaryFunction, unary_function));
    }


    // ignoring status
    template <typename Value, typename UnaryFunction>
    inline void
    transform(
      Value const& send_value, ::yampi::rank const source,
      Value& receive_value, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      ::yampi::algorithm::transform_detail::transform(
        send_value, source, receive_value, destination,
        tag, communicator, ignore_status, YAMPI_FORWARD_OR_COPY(UnaryFunction, unary_function));
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2, typename UnaryFunction>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator1>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator2>::value,
      void>::type
    transform(
      ContiguousIterator1 const send_first, ContiguousIterator1 const send_last,
      ::yampi::rank const source,
      ContiguousIterator2 const receive_first, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      ::yampi::algorithm::transform_detail::transform(
        send_first, send_last, source, receive_first, destination,
        tag, communicator, ignore_status, YAMPI_FORWARD_OR_COPY(UnaryFunction, unary_function));
    }

    template <typename ContiguousRange, typename ContiguousIterator, typename UnaryFunction>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      void>::type
    transform(
      ContiguousRange const& values, ::yampi::rank const source,
      ContiguousIterator const receive_first, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      ::yampi::algorithm::transform_detail::transform(
        boost::begin(values), boost::end(values), source, receive_first, destination,
        tag, communicator, ignore_status, YAMPI_FORWARD_OR_COPY(UnaryFunction, unary_function));
    }
  }
}


# undef YAMPI_RVALUE_REFERENCE_OR_COPY
# undef YAMPI_FORWARD_OR_COPY
# undef YAMPI_enable_if
# undef YAMPI_is_same

#endif

