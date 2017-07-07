#ifndef YAMPI_SEND_RECEIVE_HPP
# define YAMPI_SEND_RECEIVE_HPP

# include <boost/config.hpp>

/*
# include <cassert>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif
# include <iterator>
*/
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

/*
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/value_type.hpp>
*/

# include <mpi.h>

/*
# include <yampi/has_corresponding_datatype.hpp>
# include <yampi/is_contiguous_iterator.hpp>
# include <yampi/is_contiguous_range.hpp>
# include <yampi/datatype_of.hpp>
*/
//
# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
//
# include <yampi/communicator.hpp>
# include <yampi/datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/status.hpp>
# include <yampi/error.hpp>

/*
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_enable_if boost::enable_if_c
# endif
*/

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    MPI_Status stat;
    int const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::status(stat);
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    MPI_Status stat;
    int const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          const_cast<ReceiveValue*>(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::status(stat);
  }


  // with replacement
  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value>& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    MPI_Status stat;
    int const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::status(stat);
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    MPI_Status stat;
    int const error_code
      = MPI_Sendrecv_replace(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::status(stat);
  }


  // ignoring status
  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::communicator const communicator,
    ::yampi::ignore_status_t const, ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    int const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::communicator const communicator,
    ::yampi::ignore_status_t const, ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    int const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          const_cast<ReceiveValue*>(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);
  }


  // with replacement, ignoring status
  template <typename Value>
  inline void send_receive(
    ::yampi::communicator const communicator,
    ::yampi::ignore_status_t const, ::yampi::environment const& environment,
    ::yampi::buffer<Value>& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    int const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::communicator const communicator,
    ::yampi::ignore_status_t const, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    int const error_code
      = MPI_Sendrecv_replace(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);
  }
  /*
  namespace send_receive_detail
  {
    template <typename SendValue, typename ReceiveValue>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<SendValue>::value
        and ::yampi::has_corresponding_datatype<ReceiveValue>::value,
      ::yampi::status>::type
    send_receive(
      SendValue const& send_value,
      ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveValue& receive_value,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator)
    {
      MPI_Status stat;
      int const error_code
        = MPI_Sendrecv(
            const_cast<SendValue*>(YAMPI_addressof(send_value)), 1,
            ::yampi::datatype_of<SendValue>::call().mpi_datatype(),
            destination.mpi_rank(), send_tag.mpi_tag(),
            YAMPI_addressof(receive_value), 1,
            ::yampi::datatype_of<ReceiveValue>::call().mpi_datatype(),
            source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), YAMPI_addressof(stat));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::send_receive");

      return ::yampi::status(stat);
    }

    template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<
        typename std::iterator_traits<SendContiguousIterator>::value_type>::value
        and ::yampi::has_corresponding_datatype<
              typename std::iterator_traits<ReceiveContiguousIterator>::value_type>::value,
      ::yampi::status>::type
    send_receive(
      SendContiguousIterator const send_first, SendContiguousIterator const send_last,
      ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator)
    {
      assert(send_last >= send_first && receive_last >= receive_first);

      typedef
        typename std::iterator_traits<SendContiguousIterator>::value_type
        send_value_type;
      typedef
        typename std::iterator_traits<ReceiveContiguousIterator>::value_type
        receive_value_type;

      MPI_Status stat;
      int const error_code
        = MPI_Sendrecv(
            const_cast<send_value_type*>(YAMPI_addressof(*send_first)), send_last-send_first,
            ::yampi::datatype_of<send_value_type>::call().mpi_datatype(),
            destination.mpi_rank(), send_tag.mpi_tag(),
            YAMPI_addressof(*receive_first), receive_last-receive_first,
            ::yampi::datatype_of<receive_value_type>::call().mpi_datatype(),
            source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), YAMPI_addressof(stat));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::send_receive");

      return ::yampi::status(stat);
    }


    // with replacement
    template <typename Value>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<Value>::value,
      ::yampi::status>::type
    send_receive(
      Value& value,
      ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator)
    {
      MPI_Status stat;
      int const error_code
        = MPI_Sendrecv_replace(
            YAMPI_addressof(value), 1,
            ::yampi::datatype_of<Value>::call().mpi_datatype(),
            destination.mpi_rank(), send_tag.mpi_tag(),
            source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), YAMPI_addressof(stat));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::send_receive");

      return ::yampi::status(stat);
    }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<
        typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      ::yampi::status>::type
    send_receive(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator)
    {
      assert(last >= first);

      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
      MPI_Status stat;
      int const error_code
        = MPI_Sendrecv_replace(
            YAMPI_addressof(*first), last-first,
            ::yampi::datatype_of<value_type>::call().mpi_datatype(),
            destination.mpi_rank(), send_tag.mpi_tag(),
            source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), YAMPI_addressof(stat));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::send_receive");

      return ::yampi::status(stat);
    }


    // ignoring status
    template <typename SendValue, typename ReceiveValue>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<SendValue>::value
        and ::yampi::has_corresponding_datatype<ReceiveValue>::value,
      void>::type
    send_receive(
      SendValue const& send_value,
      ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveValue& receive_value,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
    {
      int const error_code
        = MPI_Sendrecv(
            const_cast<SendValue*>(YAMPI_addressof(send_value)), 1,
            ::yampi::datatype_of<SendValue>::call().mpi_datatype(),
            destination.mpi_rank(), send_tag.mpi_tag(),
            YAMPI_addressof(receive_value), 1,
            ::yampi::datatype_of<ReceiveValue>::call().mpi_datatype(),
            source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), MPI_STATUS_IGNORE);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::send_receive");
    }

    template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<
        typename std::iterator_traits<SendContiguousIterator>::value_type>::value
        and ::yampi::has_corresponding_datatype<
              typename std::iterator_traits<ReceiveContiguousIterator>::value_type>::value,
      void>::type
    send_receive(
      SendContiguousIterator const send_first, SendContiguousIterator const send_last,
      ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
    {
      assert(send_last >= send_first && receive_last >= receive_first);

      typedef
        typename std::iterator_traits<SendContiguousIterator>::value_type
        send_value_type;
      typedef
        typename std::iterator_traits<ReceiveContiguousIterator>::value_type
        receive_value_type;

      int const error_code
        = MPI_Sendrecv(
            const_cast<send_value_type*>(YAMPI_addressof(*send_first)), send_last-send_first,
            ::yampi::datatype_of<send_value_type>::call().mpi_datatype(),
            destination.mpi_rank(), send_tag.mpi_tag(),
            YAMPI_addressof(*receive_first), receive_last-receive_first,
            ::yampi::datatype_of<receive_value_type>::call().mpi_datatype(),
            source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), MPI_STATUS_IGNORE);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::send_receive");
    }


    // with replacement, ignoring status
    template <typename Value>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<Value>::value,
      void>::type
    send_receive(
      Value& value,
      ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
    {
      int const error_code
        = MPI_Sendrecv_replace(
            YAMPI_addressof(value), 1,
            ::yampi::datatype_of<Value>::call().mpi_datatype(),
            destination.mpi_rank(), send_tag.mpi_tag(),
            source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), MPI_STATUS_IGNORE);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::send_receive");
    }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    send_receive(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::rank const destination, ::yampi::tag const send_tag,
      ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
    {
      assert(last >= first);

      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;

      int const error_code
        = MPI_Sendrecv_replace(
            YAMPI_addressof(*first), last-first,
            ::yampi::datatype_of<value_type>::call().mpi_datatype(),
            destination.mpi_rank(), send_tag.mpi_tag(),
            source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), MPI_STATUS_IGNORE);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::send_receive");
    }
  } // namespace send_receive_detail


  template <typename SendValue, typename ReceiveValue>
  inline
  typename YAMPI_enable_if<
    (not ::yampi::is_contiguous_range<SendValue const>::value)
      or (not ::yampi::is_contiguous_range<ReceiveValue>::value),
    ::yampi::status>::type
  send_receive(
    SendValue const& send_value,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveValue& receive_value,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::send_receive_detail::send_receive(
      send_value, destination, send_tag,
      receive_value, source, receive_tag,
      communicator);
  }

  template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
      and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
    ::yampi::status>::type
  send_receive(
    SendContiguousIterator const send_first, SendContiguousIterator const send_last,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::send_receive_detail::send_receive(
      send_first, send_last, destination, send_tag,
      receive_first, receive_last, source, receive_tag,
      communicator);
  }

  template <typename SendContiguousRange, typename ReceiveContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<SendContiguousRange const>::value
      and ::yampi::is_contiguous_range<ReceiveContiguousRange>::value,
    ::yampi::status>::type
  send_receive(
    SendContiguousRange const& send_values,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousRange& receive_values,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::send_receive_detail::send_receive(
      boost::begin(send_values), boost::end(send_values), destination, send_tag,
      boost::begin(receive_values), boost::end(receive_values), source, receive_tag,
      communicator);
  }

  template <typename SendContiguousRange, typename ReceiveContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<SendContiguousRange const>::value
      and ::yampi::is_contiguous_range<ReceiveContiguousRange const>::value,
    ::yampi::status>::type
  send_receive(
    SendContiguousRange const& send_values,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousRange const& receive_values,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::send_receive_detail::send_receive(
      boost::begin(send_values), boost::end(send_values), destination, send_tag,
      boost::begin(receive_values), boost::end(receive_values), source, receive_tag,
      communicator);
  }


  // with replacement
  template <typename Value>
  inline
  typename YAMPI_enable_if<
    not ::yampi::is_contiguous_range<Value>::value,
    ::yampi::status>::type
  send_receive(
    Value& value,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::send_receive_detail::send_receive(
      value, destination, send_tag, source, receive_tag, communicator);
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
    ::yampi::status>::type
  send_receive(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::send_receive_detail::send_receive(
      first, last, destination, send_tag, source, receive_tag, communicator);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value,
    ::yampi::status>::type
  send_receive(
    ContiguousRange& values,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::send_receive_detail::send_receive(
      boost::begin(values), boost::end(values),
      destination, send_tag, source, receive_tag,
      communicator);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange const>::value,
    ::yampi::status>::type
  send_receive(
    ContiguousRange const& values,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::send_receive_detail::send_receive(
      boost::begin(values), boost::end(values),
      destination, send_tag, source, receive_tag,
      communicator);
  }


  // ignoring status
  template <typename SendValue, typename ReceiveValue>
  inline
  typename YAMPI_enable_if<
    (not ::yampi::is_contiguous_range<SendValue const>::value)
      or (not ::yampi::is_contiguous_range<ReceiveValue>::value),
    void>::type
  send_receive(
    SendValue const& send_value,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveValue& receive_value,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  {
    ::yampi::send_receive_detail::send_receive(
      send_value, destination, send_tag,
      receive_value, source, receive_tag,
      communicator, ignore_status);
  }

  template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
      and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
    void>::type
  send_receive(
    SendContiguousIterator const send_first, SendContiguousIterator const send_last,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  {
    ::yampi::send_receive_detail::send_receive(
      send_first, send_last, destination, send_tag,
      receive_first, receive_last, source, receive_tag,
      communicator, ignore_status);
  }

  template <typename SendContiguousRange, typename ReceiveContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<SendContiguousRange const>::value
      and ::yampi::is_contiguous_range<ReceiveContiguousRange>::value,
    void>::type
  send_receive(
    SendContiguousRange const& send_values,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousRange& receive_values,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  {
    ::yampi::send_receive_detail::send_receive(
      send_values, destination, send_tag,
      receive_values, source, receive_tag,
      communicator, ignore_status);
  }

  template <typename SendContiguousRange, typename ReceiveContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<SendContiguousRange const>::value
      and ::yampi::is_contiguous_range<ReceiveContiguousRange const>::value,
    void>::type
  send_receive(
    SendContiguousRange const& send_values,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousRange const& receive_values,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  {
    ::yampi::send_receive_detail::send_receive(
      send_values, destination, send_tag,
      receive_values, source, receive_tag,
      communicator, ignore_status);
  }


  // with replacement, ignoring status
  template <typename Value>
  inline
  typename YAMPI_enable_if<
    not ::yampi::is_contiguous_range<Value>::value,
    void>::type
  send_receive(
    Value& value,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  {
    ::yampi::send_receive_detail::send_receive(
      value, destination, send_tag, source, receive_tag, communicator, ignore_status);
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
    void>::type
  send_receive(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  {
    ::yampi::send_receive_detail::send_receive(
      first, last, destination, send_tag, source, receive_tag, communicator, ignore_status);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value,
    void>::type
  send_receive(
    ContiguousRange& values,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  {
    ::yampi::send_receive_detail::send_receive(
      boost::begin(values), boost::end(values),
      destination, send_tag,
      source, receive_tag,
      communicator, ignore_status);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange const>::value,
    void>::type
  send_receive(
    ContiguousRange const& values,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  {
    ::yampi::send_receive_detail::send_receive(
      boost::begin(values), boost::end(values),
      destination, send_tag,
      source, receive_tag,
      communicator, ignore_status);
  }
  */
}


/*
# undef YAMPI_enable_if
*/
# undef YAMPI_addressof

#endif

