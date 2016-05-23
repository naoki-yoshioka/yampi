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
# include <yampi/has_corresponding_mpi_data_type.hpp>
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
#   define YAMPI_FORWARD(Value, value) std::forward<Value>(value)
# else
#   define YAMPI_RVALUE_REFERENCE_OR_COPY(Type) Type
#   define YAMPI_FORWARD(Value, value) value
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
        ::yampi::has_corresponding_mpi_data_type<Value>::value,
        boost::optional<::yampi::status> >::type
      transform_value(
        Value const& send_value, ::yampi::rank const source, Value& receive_value, ::yampi::rank const destination,
        ::yampi::tag const tag, ::yampi::communicator const communicator,
        YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
      {
        if (source == destination)
          return boost::none;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
        auto present_rank = communicator.rank();
# else
        ::yampi::rank present_rank = communicator.rank();
# endif

        if (present_rank == destination)
          return boost::make_optional(::yampi::blocking_receive_detail::blocking_receive_value(receive_value, source, tag, communicator));

        if (present_rank == source)
          ::yampi::blocking_send_detail::blocking_send_value(unary_function(send_value), destination, tag, communicator);

        return boost::none;
      }

      template <typename ContiguousIterator1, typename ContiguousIterator2, typename UnaryFunction>
      inline
      typename YAMPI_enable_if<
        ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator1>::value_type>::value
          and YAMPI_is_same<typename std::iterator_traits<ContiguousIterator1>::type, typename std::iterator_traits<ContiguousIterator2>::type>::value,
        boost::optional<::yampi::status> >::type
      transform_iter(
        ContiguousIterator1 const send_first, int const length, ::yampi::rank const source, ContiguousIterator2 const receive_first, ::yampi::rank const destination,
        ::yampi::tag const tag, ::yampi::communicator const communicator,
        YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
      {
        if (source == destination)
          return boost::none;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
        auto present_rank = communicator.rank();
# else
        ::yampi::rank present_rank = communicator.rank();
# endif

        if (present_rank == destination)
          return boost::make_optional(::yampi::blocking_receive_detail::blocking_receive_iter(receive_first, length, source, tag, communicator));

        if (present_rank == source)
        {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
          using value_type = typename std::iterator_traits<ContiguousIterator1>::value_type;
# else
          typedef typename std::iterator_traits<ContiguousIterator1>::value_type value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
          auto buffer = std::vector<value_type>{};
#   else
          auto buffer = std::vector<value_type>();
#   endif
# else
          std::vector<value_type> buffer;
# endif
          buffer.reserve(length);
          std::transform(send_first, send_first+length, std::back_inserter(buffer), unary_function);

          ::yampi::blocking_send_detail::blocking_send_iter(buffer.begin(), length, destination, tag, communicator);
        }

        return boost::none;
      }

      template <typename ContiguousIterator1, typename ContiguousIterator2, typename UnaryFunction>
      inline
      typename YAMPI_enable_if<
        ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator1>::value_type>::value
          and YAMPI_is_same<typename std::iterator_traits<ContiguousIterator1>::type, typename std::iterator_traits<ContiguousIterator2>::type>::value,
        boost::optional<::yampi::status> >::type
      transform_iter(
        ContiguousIterator1 const send_first, ContiguousIterator1 const send_last, ::yampi::rank const source, ContiguousIterator2 const receive_first, ::yampi::rank const destination,
        ::yampi::tag const tag, ::yampi::communicator const communicator,
        YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
      {
        assert(send_last >= send_first);
        return ::yampi::algorithm::transform_detail::transform_iter(
          send_first, send_last-send_first, source, receive_first, destination,
          tag, communicator, YAMPI_FORWARD(UnaryFunction, unary_function));
      }

      template <typename ContiguousRange, typename ContiguousIterator, typename UnaryFunction>
      inline
      typename YAMPI_enable_if<
        ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value
          and YAMPI_is_same<typename boost::range_value<ContiguousRange>::type, typename std::iterator_traits<ContiguousIterator>::value_type>::value,
        boost::optional<::yampi::status> >::type
      transform_range(
        ContiguousRange const& values, ::yampi::rank const source, ContiguousIterator const receive_first, ::yampi::rank const destination,
        ::yampi::tag const tag, ::yampi::communicator const communicator,
        YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
      {
        return ::yampi::algorithm::transform_detail::transform_iter(
          boost::begin(values), boost::end(values), source, receive_first, destination,
          tag, communicator, YAMPI_FORWARD(UnaryFunction, unary_function));
      }


      // ignoring status
      template <typename Value, typename UnaryFunction>
      inline
      typename YAMPI_enable_if<::yampi::has_corresponding_mpi_data_type<Value>::value, void>::type
      transform_value(
        Value const& send_value, ::yampi::rank const source, Value& receive_value, ::yampi::rank const destination,
        ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
        YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
      {
        if (source == destination)
          return;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
        auto present_rank = communicator.rank();
# else
        ::yampi::rank present_rank = communicator.rank();
# endif

        if (present_rank == destination)
          ::yampi::blocking_receive_detail::blocking_receive_value(receive_value, source, tag, communicator, ignore_status);
        else if (present_rank == source)
          ::yampi::blocking_send_detail::blocking_send_value(unary_function(send_value), destination, tag, communicator);
      }

      template <typename ContiguousIterator1, typename ContiguousIterator2, typename UnaryFunction>
      inline
      typename YAMPI_enable_if<
        ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator1>::value_type>::value
          and YAMPI_is_same<typename std::iterator_traits<ContiguousIterator1>::type, typename std::iterator_traits<ContiguousIterator2>::type>::value,
        void>::type
      transform_iter(
        ContiguousIterator1 const send_first, int const length, ::yampi::rank const source, ContiguousIterator2 const receive_first, ::yampi::rank const destination,
        ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
        YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
      {
        if (source == destination)
          return;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
        auto present_rank = communicator.rank();
# else
        ::yampi::rank present_rank = communicator.rank();
# endif

        if (present_rank == destination)
          ::yampi::blocking_receive_detail::blocking_receive_iter(receive_first, length, source, tag, communicator, ignore_status);
        else if (present_rank == source)
        {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
          using value_type = typename std::iterator_traits<ContiguousIterator1>::value_type;
# else
          typedef typename std::iterator_traits<ContiguousIterator1>::value_type value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
          auto buffer = std::vector<value_type>{};
#   else
          auto buffer = std::vector<value_type>();
#   endif
# else
          std::vector<value_type> buffer;
# endif
          buffer.reserve(length);
          std::transform(send_first, send_first+length, std::back_inserter(buffer), unary_function);

          ::yampi::blocking_send_detail::blocking_send_iter(buffer.begin(), length, destination, tag, communicator);
        }
      }

      template <typename ContiguousIterator1, typename ContiguousIterator2, typename UnaryFunction>
      inline
      typename YAMPI_enable_if<
        ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator1>::value_type>::value
          and YAMPI_is_same<typename std::iterator_traits<ContiguousIterator1>::type, typename std::iterator_traits<ContiguousIterator2>::type>::value,
        void>::type
      transform_iter(
        ContiguousIterator1 const send_first, ContiguousIterator1 const send_last, ::yampi::rank const source, ContiguousIterator2 const receive_first, ::yampi::rank const destination,
        ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
        YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
      {
        assert(send_last >= send_first);
        ::yampi::algorithm::transform_detail::transform_iter(
          send_first, send_last-send_first, source, receive_first, destination,
          tag, communicator, ignore_status, YAMPI_FORWARD(UnaryFunction, unary_function));
      }

      template <typename ContiguousRange, typename ContiguousIterator, typename UnaryFunction>
      inline
      typename YAMPI_enable_if<
        ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value
          and YAMPI_is_same<typename boost::range_value<ContiguousRange>::type, typename std::iterator_traits<ContiguousIterator>::value_type>::value,
        void>::type
      transform_range(
        ContiguousRange const& values, ::yampi::rank const source, ContiguousIterator const receive_first, ::yampi::rank const destination,
        ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
        YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
      {
        ::yampi::algorithm::transform_detail::transform_iter(
          boost::begin(values), boost::end(values), source, receive_first, destination,
          tag, communicator, ignore_status, YAMPI_FORWARD(UnaryFunction, unary_function));
      }
    } // namespace transform_detail


    template <typename Value, typename UnaryFunction>
    inline boost::optional<::yampi::status>
    transform(
      Value const& send_value, ::yampi::rank const source,
      Value& receive_value, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      return ::yampi::algorithm::transform_detail::transform_value(
        send_value, source, receive_value, destination, tag, communicator, YAMPI_FORWARD(UnaryFunction, unary_function));
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2, typename UnaryFunction>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator1>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator2>::value,
      boost::optional<::yampi::status> >::type
    transform(
      ContiguousIterator1 const send_first, int const length, ::yampi::rank const source,
      ContiguousIterator2 const receive_first, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      return ::yampi::algorithm::transform_detail::transform_iter(
        send_first, length, source, receive_first, destination,
        tag, communicator, YAMPI_FORWARD(UnaryFunction, unary_function));
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2, typename UnaryFunction>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator1>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator2>::value,
      boost::optional<::yampi::status> >::type
    transform(
      ContiguousIterator1 const send_first, ContiguousIterator1 const send_last, ::yampi::rank const source,
      ContiguousIterator2 const receive_first, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      return ::yampi::algorithm::transform_detail::transform_iter(
        send_first, send_last, source, receive_first, destination,
        tag, communicator, YAMPI_FORWARD(UnaryFunction, unary_function));
    }

    template <typename ContiguousRange, typename ContiguousIterator, typename UnaryFunction>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      boost::optional<::yampi::status> >::type
    transform(
      ContiguousRange const& values, ::yampi::rank const source,
      ContiguousIterator const receive_first, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      return ::yampi::algorithm::transform_detail::transform_range(
        values, source, receive_first, destination,
        tag, communicator, YAMPI_FORWARD(UnaryFunction, unary_function));
    }


    // ignoring status
    template <typename Value, typename UnaryFunction>
    inline void
    transform(
      Value const& send_value, ::yampi::rank const source,
      Value& receive_value, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      ::yampi::algorithm::transform_detail::transform_value(
        send_value, source, receive_value, destination,
        tag, communicator, ignore_status, YAMPI_FORWARD(UnaryFunction, unary_function));
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2, typename UnaryFunction>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator1>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator2>::value,
      void>::type
    transform(
      ContiguousIterator1 const send_first, int const length, ::yampi::rank const source,
      ContiguousIterator2 const receive_first, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      ::yampi::algorithm::transform_detail::transform_iter(
        send_first, length, source, receive_first, destination,
        tag, communicator, ignore_status, YAMPI_FORWARD(UnaryFunction, unary_function));
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2, typename UnaryFunction>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator1>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator2>::value,
      void>::type
    transform(
      ContiguousIterator1 const send_first, ContiguousIterator1 const send_last, ::yampi::rank const source,
      ContiguousIterator2 const receive_first, ::yampi::rank const destination,
      ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      ::yampi::algorithm::transform_detail::transform_iter(
        send_first, send_last, source, receive_first, destination,
        tag, communicator, ignore_status, YAMPI_FORWARD(UnaryFunction, unary_function));
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
      ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      YAMPI_RVALUE_REFERENCE_OR_COPY(UnaryFunction) unary_function)
    {
      ::yampi::algorithm::transform_detail::transform_range(
        values, source, receive_first, destination,
        tag, communicator, ignore_status, YAMPI_FORWARD(UnaryFunction, unary_function));
    }
  }
}


# undef YAMPI_RVALUE_REFERENCE_OR_COPY
# undef YAMPI_FORWARD
# undef YAMPI_enable_if
# undef YAMPI_is_same

#endif

