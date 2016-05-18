#ifndef YAMPI_SEND_RECEIVE_HPP
# define YAMPI_SEND_RECEIVE_HPP

# include <boost/config.hpp>

# include <cassert>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif
# include <iterator>

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/value_type.hpp>

# include <mpi.h>

# include <yampi/has_corresponding_mpi_data_type.hpp>
# include <yampi/is_contiguous_iterator.hpp>
# include <yampi/is_contiguous_range.hpp>
# include <yampi/mpi_data_type_of.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/status.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_enable_if boost::enable_if
# endif


namespace yampi
{
  template <typename SendValue, typename ReceiveValue>
  inline
  typename YAMPI_enable_if<
    ::yampi::has_corresponding_mpi_data_type<SendValue>::value
      and ::yampi::has_corresponding_mpi_data_type<ReceiveValue>::value,
    ::yampi::status<ReceiveValue> >::type
  send_receive(
    SendValue const& send_value, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveValue& receive_value, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto stat = MPI_Status{};
#   else
    auto stat = MPI_Status();
#   endif

    auto const error_code
      = MPI_Sendrecv(
          &send_value, 1, ::yampi::mpi_data_type_of<SendValue>::value, destination.mpi_rank(), send_tag.mpi_tag(),
          &receive_value, 1, ::yampi::mpi_data_type_of<ReceiveValue>::value, source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), &stat);
# else
    MPI_Status stat;

    int const error_code
      = MPI_Sendrecv(
          &send_value, 1, ::yampi::mpi_data_type_of<SendValue>::value, destination.mpi_rank(), send_tag.mpi_tag(),
          &receive_value, 1, ::yampi::mpi_data_type_of<ReceiveValue>::value, source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), &stat);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive"};

    return ::yampi::status{stat};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive");

    return ::yampi::status(stat);
# endif
  }

  template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
      and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<SendContiguousIterator>::value_type>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ReceiveContiguousIterator>::value_type>::value,
    ::yampi::status<typename std::iterator_traits<ReceiveContiguousIterator>::value_type> >::type
  send_receive(
    SendContiguousIterator const send_first, int const send_length, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousIterator const receive_first, int const receivelength, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    typedef typename std::iterator_traits<SendContiguousIterator>::value_type send_value_type;
    typedef typename std::iterator_traits<ReceiveContiguousIterator>::value_type receive_value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto stat = MPI_Status{};
#   else
    auto stat = MPI_Status();
#   endif

    auto const error_code
      = MPI_Sendrecv(
          &*send_first, send_length, ::yampi::mpi_data_type_of<send_value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(),
          &*receive_first, receive_length, ::yampi::mpi_data_type_of<receive_value_type>::value, source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), &stat);
# else
    MPI_Status stat;

    int const error_code
      = MPI_Sendrecv(
          &*send_first, send_length, ::yampi::mpi_data_type_of<send_value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(),
          &*receive_first, receive_length, ::yampi::mpi_data_type_of<receive_value_type>::value, source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), &stat);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive"};

    return ::yampi::status{stat};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive");

    return ::yampi::status(stat);
# endif
  }

  template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
      and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<SendContiguousIterator>::value_type>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ReceiveContiguousIterator>::value_type>::value,
    ::yampi::status<typename std::iterator_traits<ReceiveiguousIterator>::value_type> >::type
  send_receive(
    SendContiguousIterator const send_first, SendContiguousIterator const send_last, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    assert(send_last >= send_first && receive_last >= receive_first);
    return send_receive(
      send_first, send_last-send_first, destination, send_tag,
      receive_first, receive_last-receive_first, source, receive_tag,
      communicator);
  }

  template <typename SendContiguousRange, typename ReceiveContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<SendContiguousRange>::value
      and ::yampi::is_contiguous_range<ReceiveContiguousRange>::value
      and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<SendContiguousRange>::type>::value
      and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ReceiveContiguousRange>::type>::value,
    ::yampi::status<typename boost::range_value<ReceiveContiguousRange>::type> >::type
  send_receive(
    SendContiguousRange const& send_values, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousRange& receive_values, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    return send_receive(
      boost::begin(send_values), boost::end(send_values), destination, send_tag,
      boost::begin(receive_values), boost::end(receive_values), source, receive_tag,
      communicator);
  }


  // with replacement
  template <typename Value>
  inline
  typename YAMPI_enable_if<
    ::yampi::has_corresponding_mpi_data_type<Value>::value,
    ::yampi::status<Value> >::type
  send_receive(
    Value& value, ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto stat = MPI_Status{};
#   else
    auto stat = MPI_Status();
#   endif

    auto const error_code
      = MPI_Sendrecv_replace(&value, 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), &stat);
# else
    MPI_Status stat;

    int const error_code
      = MPI_Sendrecv_replace(&value, 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), &stat);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive"};

    return ::yampi::status{stat};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive");

    return ::yampi::status(stat);
# endif
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    ::yampi::status<typename std::iterator_traits<ContiguousIterator>::value_type> >::type
  send_receive(
    ContiguousIterator const first, int const length,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto stat = MPI_Status{};
#   else
    auto stat = MPI_Status();
#   endif

    auto const error_code
      = MPI_Sendrecv_replace(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), &stat);
# else
    MPI_Status stat;

    int const error_code
      = MPI_Sendrecv_replace(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), &stat);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive"};

    return ::yampi::status{stat};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive");

    return ::yampi::status(stat);
# endif
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    ::yampi::status<typename std::iterator_traits<ContiguousIterator>::value_type> >::type
  send_receive(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  {
    assert(last >= first);
    return send_receive(first, last-first, destination, send_tag, source, receive_tag, communicator);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value
      and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
    ::yampi::status<typename boost::range_value<ContiguousRange>::type> >::type
  send_receive(
    ContiguousRange& values,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  { return send_receive(boost::begin(values), boost::end(values), destination, send_tag, source, receive_tag, communicator); }


  // ignoring status
  template <typename SendValue, typename ReceiveValue>
  inline
  typename YAMPI_enable_if<
    ::yampi::has_corresponding_mpi_data_type<SendValue>::value
      and ::yampi::has_corresponding_mpi_data_type<ReceiveValue>::value,
    void>::type
  send_receive(
    SendValue const& send_value, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveValue& receive_value, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
    auto const error_code
      = MPI_Sendrecv(
          &send_value, 1, ::yampi::mpi_data_type_of<SendValue>::value, destination.mpi_rank(), send_tag.mpi_tag(),
          &receive_value, 1, ::yampi::mpi_data_type_of<ReceiveValue>::value, source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
    int const error_code
      = MPI_Sendrecv(
          &send_value, 1, ::yampi::mpi_data_type_of<SendValue>::value, destination.mpi_rank(), send_tag.mpi_tag(),
          &receive_value, 1, ::yampi::mpi_data_type_of<ReceiveValue>::value, source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive");
# endif
  }

  template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
      and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<SendContiguousIterator>::value_type>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ReceiveContiguousIterator>::value_type>::value,
    void>::type
  send_receive(
    SendContiguousIterator const send_first, int const send_length, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousIterator const receive_first, int const receivelength, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
  {
    typedef typename std::iterator_traits<SendContiguousIterator>::value_type send_value_type;
    typedef typename std::iterator_traits<ReceiveContiguousIterator>::value_type receive_value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
    auto const error_code
      = MPI_Sendrecv(
          &*send_first, send_length, ::yampi::mpi_data_type_of<send_value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(),
          &*receive_first, receive_length, ::yampi::mpi_data_type_of<receive_value_type>::value, source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
    int const error_code
      = MPI_Sendrecv(
          &*send_first, send_length, ::yampi::mpi_data_type_of<send_value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(),
          &*receive_first, receive_length, ::yampi::mpi_data_type_of<receive_value_type>::value, source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive");
# endif
  }

  template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
      and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<SendContiguousIterator>::value_type>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ReceiveContiguousIterator>::value_type>::value,
    void>::type
  send_receive(
    SendContiguousIterator const send_first, SendContiguousIterator const send_last, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  {
    assert(send_last >= send_first && receive_last >= receive_first);
    ::yampi::send_receive(
      send_first, send_last-send_first, destination, send_tag,
      receive_first, receive_last-receive_first, source, receive_tag,
      communicator, ignore_status);
  }

  template <typename SendContiguousRange, typename ReceiveContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<SendContiguousRange>::value
      and ::yampi::is_contiguous_range<ReceiveContiguousRange>::value
      and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<SendContiguousRange>::type>::value
      and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ReceiveContiguousRange>::type>::value,
    void>::type
  send_receive(
    SendContiguousRange const& send_values, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousRange& receive_values, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  {
    ::yampi::send_receive(
      boost::begin(send_values), boost::end(send_values), destination, send_tag,
      boost::begin(receive_values), boost::end(receive_values), source, receive_tag,
      communicator, ignore_status);
  }


  // with replacement, ignoring status
  template <typename Value>
  inline
  typename YAMPI_enable_if<
    ::yampi::has_corresponding_mpi_data_type<Value>::value,
    ::yampi::status<Value> >::type
  send_receive(
    Value& value, ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
    auto const error_code
      = MPI_Sendrecv_replace(&value, 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
    int const error_code
      = MPI_Sendrecv_replace(&value, 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive");
# endif
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    void>::type
  send_receive(
    ContiguousIterator const first, int const length,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
  {
    typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
    auto const error_code
      = MPI_Sendrecv_replace(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
    int const error_code
      = MPI_Sendrecv_replace(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive");
# endif
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    void>::type
  send_receive(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  {
    assert(last >= first);
    ::yampi::send_receive(first, last-first, destination, send_tag, source, receive_tag, communicator, ignore_status);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value
      and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
    void>::type
  send_receive(
    ContiguousRange& values,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::send_receive(boost::begin(values), boost::end(values), destination, send_tag, source, receive_tag, communicator, ignore_status); }
}


# undef YAMPI_enable_if

#endif

