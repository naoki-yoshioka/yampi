#ifndef YAMPI_ALGORITHM_SWAP_HPP
# define YAMPI_ALGORITHM_SWAP_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif
# include <iterator>

# include <boost/range/value_type.hpp>

# include <yampi/send_receive.hpp>
# include <yampi/status.hpp>
# include <yampi/tag.hpp>
# include <yampi/rank.hpp>
# include <yampi/communicator.hpp>
# include <yampi/is_contiguous_iterator.hpp>
# include <yampi/is_contiguous_range.hpp>
# include <yampi/communicator.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_enable_if boost::enable_if_c
# endif


namespace yampi
{
  namespace algorithm
  {
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    template <typename SendValue, typename ReceiveValue>
    inline
    typename YAMPI_enable_if<
      (not ::yampi::is_contiguous_range<SendValue const>::value)
        or (not ::yampi::is_contiguous_range<ReceiveValue>::value),
      ::yampi::status>::type
    swap(
      SendValue const& send_value, ReceiveValue& receive_value,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::tag const tag = ::yampi::tag{0})
    {
      return ::yampi::send_receive_detail::send_receive_value(
        send_value, swap_rank, tag, receive_value, swap_rank, tag, communicator);
    }

    template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
        and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
      ::yampi::status>::type
    swap(
      SendContiguousIterator const send_first, int const send_length,
      ReceiveContiguousIterator const receive_first, int const receive_length,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::tag const tag = ::yampi::tag{0})
    {
      return ::yampi::send_receive_detail::send_receive_iter(
        send_first, send_length, swap_rank, tag,
        receive_first, receive_length, swap_rank, tag, communicator);
    }

    template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
        and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
      ::yampi::status>::type
    swap(
      SendContiguousIterator const send_first, SendContiguousIterator const send_last,
      ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::tag const tag = ::yampi::tag{0})
    {
      return ::yampi::send_receive_detail::send_receive_iter(
        send_first, send_last, swap_rank, tag,
        receive_first, receive_last, swap_rank, tag, communicator);
    }

    template <typename SendContiguousRange, typename ReceiveContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<SendContiguousRange const>::value
        and ::yampi::is_contiguous_range<ReceiveContiguousRange>::value,
      ::yampi::status>::type
    swap(
      SendContiguousRange const& send_values, ReceiveContiguousRange& receive_values,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::tag const tag = ::yampi::tag{0})
    {
      return ::yampi::send_receive_detail::send_receive_range(
        send_values, swap_rank, tag, receive_values, swap_rank, tag, communicator);
    }

    template <typename SendContiguousRange, typename ReceiveContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<SendContiguousRange const>::value
        and ::yampi::is_contiguous_range<ReceiveContiguousRange const>::value,
      ::yampi::status>::type
    swap(
      SendContiguousRange const& send_values, ReceiveContiguousRange const& receive_values,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::tag const tag = ::yampi::tag{0})
    {
      return ::yampi::send_receive_detail::send_receive_range(
        send_values, swap_rank, tag, receive_values, swap_rank, tag, communicator);
    }


    // with replacement
    template <typename Value>
    inline
    typename YAMPI_enable_if<not ::yampi::is_contiguous_range<Value>::value, ::yampi::status>::type
    swap(
      Value& value, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::tag const tag = ::yampi::tag{0})
    { return ::yampi::send_receive_detail::send_receive_value(value, swap_rank, tag, swap_rank, tag, communicator); }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, ::yampi::status>::type
    swap(
      ContiguousIterator const first, int const length, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::tag const tag = ::yampi::tag{0})
    { return ::yampi::send_receive_detail::send_receive_iter(first, length, swap_rank, tag, swap_rank, tag, communicator); }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, ::yampi::status>::type
    swap(
      ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::tag const tag = ::yampi::tag{0})
    { return ::yampi::send_receive_detail::send_receive_iter(first, last, swap_rank, tag, swap_rank, tag, communicator); }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange>::value, ::yampi::status>::type
    swap(
      ContiguousRange& values, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::tag const tag = ::yampi::tag{0})
    {
      return ::yampi::send_receive_detail::send_receive_range(
        values, swap_rank, tag, swap_rank, tag, communicator);
    }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange const>::value, ::yampi::status>::type
    swap(
      ContiguousRange const& values, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::tag const tag = ::yampi::tag{0})
    {
      return ::yampi::send_receive_detail::send_receive_range(
        values, swap_rank, tag, swap_rank, tag, communicator);
    }


    // ignoring status
    template <typename SendValue, typename ReceiveValue>
    inline
    typename YAMPI_enable_if<
      (not ::yampi::is_contiguous_range<SendValue const>::value)
        or (not ::yampi::is_contiguous_range<ReceiveValue>::value),
      void>::type
    swap(
      SendValue const& send_value, ReceiveValue& receive_value,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status, ::yampi::tag const tag = ::yampi::tag{0})
    {
      ::yampi::send_receive_detail::send_receive_value(
        send_value, swap_rank, tag, receive_value, swap_rank, tag, communicator, ignore_status);
    }

    template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
        and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
      void>::type
    swap(
      SendContiguousIterator const send_first, int const send_length,
      ReceiveContiguousIterator const receive_first, int const receive_length,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status, ::yampi::tag const tag = ::yampi::tag{0})
    {
      ::yampi::send_receive_detail::send_receive_iter(
        send_first, send_length, swap_rank, tag,
        receive_first, receive_length, swap_rank, tag, communicator, ignore_status);
    }

    template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
        and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
      void>::type
    swap(
      SendContiguousIterator const send_first, SendContiguousIterator const send_last,
      ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status, ::yampi::tag const tag = ::yampi::tag{0})
    {
      ::yampi::send_receive_detail::send_receive_iter(
        send_first, send_last, swap_rank, tag,
        receive_first, receive_last, swap_rank, tag, communicator, ignore_status);
    }

    template <typename SendContiguousRange, typename ReceiveContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<SendContiguousRange const>::value
        and ::yampi::is_contiguous_range<ReceiveContiguousRange>::value,
      void>::type
    swap(
      SendContiguousRange const& send_values, ReceiveContiguousRange& receive_values,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status, ::yampi::tag const tag = ::yampi::tag{0})
    {
      ::yampi::send_receive_detail::send_receive_range(
        send_values, swap_rank, tag, receive_values, swap_rank, tag, communicator, ignore_status);
    }

    template <typename SendContiguousRange, typename ReceiveContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<SendContiguousRange const>::value
        and ::yampi::is_contiguous_range<ReceiveContiguousRange const>::value,
      void>::type
    swap(
      SendContiguousRange const& send_values, ReceiveContiguousRange const& receive_values,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status, ::yampi::tag const tag = ::yampi::tag{0})
    {
      ::yampi::send_receive_detail::send_receive_range(
        send_values, swap_rank, tag, receive_values, swap_rank, tag, communicator, ignore_status);
    }


    // with replacement, ignoring status
    template <typename Value>
    inline
    typename YAMPI_enable_if<not ::yampi::is_contiguous_range<Value>::value, void>::type
    swap(
      Value& value, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      ::yampi::tag const tag = ::yampi::tag{0})
    { ::yampi::send_receive_detail::send_receive_value(value, swap_rank, tag, swap_rank, tag, communicator, ignore_status); }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, void>::type
    swap(
      ContiguousIterator const first, int const length, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      ::yampi::tag const tag = ::yampi::tag{0})
    { ::yampi::send_receive_detail::send_receive_iter(first, length, swap_rank, tag, swap_rank, tag, communicator, ignore_status); }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, void>::type
    swap(
      ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      ::yampi::tag const tag = ::yampi::tag{0})
    { ::yampi::send_receive_detail::send_receive_iter(first, last, swap_rank, tag, swap_rank, tag, communicator, ignore_status); }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange>::value, void>::type
    swap(
      ContiguousRange& values, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      ::yampi::tag const tag = ::yampi::tag{0})
    {
      ::yampi::send_receive_detail::send_receive_range(
        values, swap_rank, tag, swap_rank, tag, communicator, ignore_status);
    }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange const>::value, void>::type
    swap(
      ContiguousRange const& values, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      ::yampi::tag const tag = ::yampi::tag{0})
    {
      ::yampi::send_receive_detail::send_receive_range(
        values, swap_rank, tag, swap_rank, tag, communicator, ignore_status);
    }
# else // BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    template <typename SendValue, typename ReceiveValue>
    inline
    typename YAMPI_enable_if<
      (not ::yampi::is_contiguous_range<SendValue const>::value)
        or (not ::yampi::is_contiguous_range<ReceiveValue>::value),
      ::yampi::status>::type
    swap(
      SendValue const& send_value, ReceiveValue& receive_value,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      return ::yampi::send_receive_detail::send_receive_value(
        send_value, swap_rank, tag, receive_value, swap_rank, tag, communicator);
    }

    template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
        and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
      ::yampi::status>::type
    swap(
      SendContiguousIterator const send_first, int const send_length,
      ReceiveContiguousIterator const receive_first, int const receive_length,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      return ::yampi::send_receive_detail::send_receive_iter(
        send_first, send_length, swap_rank, tag,
        receive_first, receive_length, swap_rank, tag, communicator);
    }

    template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
        and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
      ::yampi::status>::type
    swap(
      SendContiguousIterator const send_first, SendContiguousIterator const send_last,
      ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      return ::yampi::send_receive_detail::send_receive_iter(
        send_first, send_last, swap_rank, tag,
        receive_first, receive_last, swap_rank, tag, communicator);
    }

    template <typename SendContiguousRange, typename ReceiveContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<SendContiguousRange const>::value
        and ::yampi::is_contiguous_range<ReceiveContiguousRange>::value,
      ::yampi::status>::type
    swap(
      SendContiguousRange const& send_values, ReceiveContiguousRange& receive_values,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      return ::yampi::send_receive_detail::send_receive_range(
        send_values, swap_rank, tag, receive_values, swap_rank, tag, communicator);
    }

    template <typename SendContiguousRange, typename ReceiveContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<SendContiguousRange const>::value
        and ::yampi::is_contiguous_range<ReceiveContiguousRange const>::value,
      ::yampi::status>::type
    swap(
      SendContiguousRange const& send_values, ReceiveContiguousRange const& receive_values,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      return ::yampi::send_receive_detail::send_receive_range(
        send_values, swap_rank, tag, receive_values, swap_rank, tag, communicator);
    }


    // with replacement
    template <typename Value>
    inline
    typename YAMPI_enable_if<not ::yampi::is_contiguous_range<Value>::value, ::yampi::status>::type
    swap(
      Value& value, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::tag const tag = ::yampi::tag(0))
    { return ::yampi::send_receive_detail::send_receive_value(value, swap_rank, tag, swap_rank, tag, communicator); }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, ::yampi::status>::type
    swap(
      ContiguousIterator const first, int const length, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::tag const tag = ::yampi::tag(0))
    { return ::yampi::send_receive_detail::send_receive_iter(first, length, swap_rank, tag, swap_rank, tag, communicator); }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, ::yampi::status>::type
    swap(
      ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::tag const tag = ::yampi::tag(0))
    { return ::yampi::send_receive_detail::send_receive_iter(first, last, swap_rank, tag, swap_rank, tag, communicator); }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange>::value, ::yampi::status>::type
    swap(
      ContiguousRange& values, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::tag const tag = ::yampi::tag(0))
    {
      return ::yampi::send_receive_detail::send_receive_range(
        values, swap_rank, tag, swap_rank, tag, communicator);
    }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange const>::value, ::yampi::status>::type
    swap(
      ContiguousRange const& values, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::tag const tag = ::yampi::tag(0))
    {
      return ::yampi::send_receive_detail::send_receive_range(
        values, swap_rank, tag, swap_rank, tag, communicator);
    }


    // ignoring status
    template <typename SendValue, typename ReceiveValue>
    inline
    typename YAMPI_enable_if<
      (not ::yampi::is_contiguous_range<SendValue const>::value)
        or (not ::yampi::is_contiguous_range<ReceiveValue>::value),
      void>::type
    swap(
      SendValue const& send_value, ReceiveValue& receive_value,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status, ::yampi::tag const tag = ::yampi::tag(0))
    {
      ::yampi::send_receive_detail::send_receive_value(
        send_value, swap_rank, tag, receive_value, swap_rank, tag, communicator, ignore_status);
    }

    template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
        and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
      void>::type
    swap(
      SendContiguousIterator const send_first, int const send_length,
      ReceiveContiguousIterator const receive_first, int const receive_length,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status, ::yampi::tag const tag = ::yampi::tag(0))
    {
      ::yampi::send_receive_detail::send_receive_iter(
        send_first, send_length, swap_rank, tag,
        receive_first, receive_length, swap_rank, tag, communicator, ignore_status);
    }

    template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
        and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
      void>::type
    swap(
      SendContiguousIterator const send_first, SendContiguousIterator const send_last,
      ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status, ::yampi::tag const tag = ::yampi::tag(0))
    {
      ::yampi::send_receive_detail::send_receive_iter(
        send_first, send_last, swap_rank, tag,
        receive_first, receive_last, swap_rank, tag, communicator, ignore_status);
    }

    template <typename SendContiguousRange, typename ReceiveContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<SendContiguousRange const>::value
        and ::yampi::is_contiguous_range<ReceiveContiguousRange>::value,
      void>::type
    swap(
      SendContiguousRange const& send_values, ReceiveContiguousRange& receive_values,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status, ::yampi::tag const tag = ::yampi::tag(0))
    {
      ::yampi::send_receive_detail::send_receive_range(
        send_values, swap_rank, tag, receive_values, swap_rank, tag, communicator, ignore_status);
    }

    template <typename SendContiguousRange, typename ReceiveContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<SendContiguousRange const>::value
        and ::yampi::is_contiguous_range<ReceiveContiguousRange const>::value,
      void>::type
    swap(
      SendContiguousRange const& send_values, ReceiveContiguousRange const& receive_values,
      ::yampi::rank const swap_rank, ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const ignore_status, ::yampi::tag const tag = ::yampi::tag(0))
    {
      ::yampi::send_receive_detail::send_receive_range(
        send_values, swap_rank, tag, receive_values, swap_rank, tag, communicator, ignore_status);
    }


    // with replacement, ignoring status
    template <typename Value>
    inline
    typename YAMPI_enable_if<not ::yampi::is_contiguous_range<Value>::value, void>::type
    swap(
      Value& value, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      ::yampi::tag const tag = ::yampi::tag(0))
    { ::yampi::send_receive_detail::send_receive_value(value, swap_rank, tag, swap_rank, tag, communicator, ignore_status); }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, void>::type
    swap(
      ContiguousIterator const first, int const length, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      ::yampi::tag const tag = ::yampi::tag(0))
    { ::yampi::send_receive_detail::send_receive_iter(first, length, swap_rank, tag, swap_rank, tag, communicator, ignore_status); }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, void>::type
    swap(
      ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      ::yampi::tag const tag = ::yampi::tag(0))
    { ::yampi::send_receive_detail::send_receive_iter(first, last, swap_rank, tag, swap_rank, tag, communicator, ignore_status); }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange>::value, void>::type
    swap(
      ContiguousRange& values, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      ::yampi::send_receive_detail::send_receive_range(
        values, swap_rank, tag, swap_rank, tag, communicator, ignore_status);
    }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange const>::value, void>::type
    swap(
      ContiguousRange const& values, ::yampi::rank const swap_rank,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      ::yampi::send_receive_detail::send_receive_range(
        values, swap_rank, tag, swap_rank, tag, communicator, ignore_status);
    }
# endif // BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
  }
}


# undef YAMPI_enable_if

#endif

