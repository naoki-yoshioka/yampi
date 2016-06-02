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
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

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
#   define YAMPI_enable_if boost::enable_if_c
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  namespace send_receive_detail
  {
    template <typename SendValue, typename ReceiveValue>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<SendValue>::value
        and ::yampi::has_corresponding_mpi_data_type<ReceiveValue>::value,
      ::yampi::status>::type
    send_receive_value(
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
            YAMPI_addressof(send_value), 1, ::yampi::mpi_data_type_of<SendValue>::value, destination.mpi_rank(), send_tag.mpi_tag(),
            YAMPI_addressof(receive_value), 1, ::yampi::mpi_data_type_of<ReceiveValue>::value, source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), YAMPI_addressof(stat));
# else
      MPI_Status stat;

      int const error_code
        = MPI_Sendrecv(
            YAMPI_addressof(send_value), 1, ::yampi::mpi_data_type_of<SendValue>::value, destination.mpi_rank(), send_tag.mpi_tag(),
            YAMPI_addressof(receive_value), 1, ::yampi::mpi_data_type_of<ReceiveValue>::value, source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), YAMPI_addressof(stat));
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
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<SendContiguousIterator>::value_type>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ReceiveContiguousIterator>::value_type>::value,
      ::yampi::status>::type
    send_receive_iter(
      SendContiguousIterator const send_first, int const send_length, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveContiguousIterator const receive_first, int const receive_length, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator)
    {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
      using send_value_type = typename std::iterator_traits<SendContiguousIterator>::value_type;
      using receive_value_type = typename std::iterator_traits<ReceiveContiguousIterator>::value_type;
# else
      typedef typename std::iterator_traits<SendContiguousIterator>::value_type send_value_type;
      typedef typename std::iterator_traits<ReceiveContiguousIterator>::value_type receive_value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto stat = MPI_Status{};
#   else
      auto stat = MPI_Status();
#   endif

      auto const error_code
        = MPI_Sendrecv(
            YAMPI_addressof(*send_first), send_length, ::yampi::mpi_data_type_of<send_value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(),
            YAMPI_addressof(*receive_first), receive_length, ::yampi::mpi_data_type_of<receive_value_type>::value, source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), YAMPI_addressof(stat));
# else
      MPI_Status stat;

      int const error_code
        = MPI_Sendrecv(
            YAMPI_addressof(*send_first), send_length, ::yampi::mpi_data_type_of<send_value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(),
            YAMPI_addressof(*receive_first), receive_length, ::yampi::mpi_data_type_of<receive_value_type>::value, source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), YAMPI_addressof(stat));
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
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<SendContiguousIterator>::value_type>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ReceiveContiguousIterator>::value_type>::value,
      ::yampi::status>::type
    send_receive_iter(
      SendContiguousIterator const send_first, SendContiguousIterator const send_last, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator)
    {
      assert(send_last >= send_first && receive_last >= receive_first);
      return ::yampi::send_receive_detail::send_receive_iter(
        send_first, send_last-send_first, destination, send_tag,
        receive_first, receive_last-receive_first, source, receive_tag,
        communicator);
    }

    template <typename SendContiguousRange, typename ReceiveContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<SendContiguousRange const>::type>::value
        and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ReceiveContiguousRange>::type>::value,
      ::yampi::status>::type
    send_receive_range(
      SendContiguousRange const& send_values, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveContiguousRange& receive_values, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator)
    {
      return ::yampi::send_receive_detail::send_receive_iter(
        boost::begin(send_values), boost::end(send_values), destination, send_tag,
        boost::begin(receive_values), boost::end(receive_values), source, receive_tag,
        communicator);
    }

    template <typename SendContiguousRange, typename ReceiveContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<SendContiguousRange const>::type>::value
        and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ReceiveContiguousRange const>::type>::value,
      ::yampi::status>::type
    send_receive_range(
      SendContiguousRange const& send_values, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveContiguousRange const& receive_values, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator)
    {
      return ::yampi::send_receive_detail::send_receive_iter(
        boost::begin(send_values), boost::end(send_values), destination, send_tag,
        boost::begin(receive_values), boost::end(receive_values), source, receive_tag,
        communicator);
    }


    // with replacement
    template <typename Value>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<Value>::value,
      ::yampi::status>::type
    send_receive_value(
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
        = MPI_Sendrecv_replace(YAMPI_addressof(value), 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(stat));
# else
      MPI_Status stat;

      int const error_code
        = MPI_Sendrecv_replace(YAMPI_addressof(value), 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(stat));
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
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      ::yampi::status>::type
    send_receive_iter(
      ContiguousIterator const first, int const length,
      ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator)
    {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
      using value_type = typename std::iterator_traits<ContiguousIterator>::value_type;
# else
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto stat = MPI_Status{};
#   else
      auto stat = MPI_Status();
#   endif

      auto const error_code
        = MPI_Sendrecv_replace(YAMPI_addressof(*first), length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(stat));
# else
      MPI_Status stat;

      int const error_code
        = MPI_Sendrecv_replace(YAMPI_addressof(*first), length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(stat));
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
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      ::yampi::status>::type
    send_receive_iter(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator)
    {
      assert(last >= first);
      return ::yampi::send_receive_detail::send_receive_iter(first, last-first, destination, send_tag, source, receive_tag, communicator);
    }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
      ::yampi::status>::type
    send_receive_range(
      ContiguousRange& values,
      ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator)
    { return ::yampi::send_receive_detail::send_receive_iter(boost::begin(values), boost::end(values), destination, send_tag, source, receive_tag, communicator); }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange const>::type>::value,
      ::yampi::status>::type
    send_receive_range(
      ContiguousRange const& values,
      ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator)
    { return ::yampi::send_receive_detail::send_receive_iter(boost::begin(values), boost::end(values), destination, send_tag, source, receive_tag, communicator); }


    // ignoring status
    template <typename SendValue, typename ReceiveValue>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<SendValue>::value
        and ::yampi::has_corresponding_mpi_data_type<ReceiveValue>::value,
      void>::type
    send_receive_value(
      SendValue const& send_value, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveValue& receive_value, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Sendrecv(
            YAMPI_addressof(send_value), 1, ::yampi::mpi_data_type_of<SendValue>::value, destination.mpi_rank(), send_tag.mpi_tag(),
            YAMPI_addressof(receive_value), 1, ::yampi::mpi_data_type_of<ReceiveValue>::value, source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
      int const error_code
        = MPI_Sendrecv(
            YAMPI_addressof(send_value), 1, ::yampi::mpi_data_type_of<SendValue>::value, destination.mpi_rank(), send_tag.mpi_tag(),
            YAMPI_addressof(receive_value), 1, ::yampi::mpi_data_type_of<ReceiveValue>::value, source.mpi_rank(), receive_tag.mpi_tag(),
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
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<SendContiguousIterator>::value_type>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ReceiveContiguousIterator>::value_type>::value,
      void>::type
    send_receive_iter(
      SendContiguousIterator const send_first, int const send_length, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveContiguousIterator const receive_first, int const receive_length, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
    {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
      using send_value_type = typename std::iterator_traits<SendContiguousIterator>::value_type;
      using receive_value_type = typename std::iterator_traits<ReceiveContiguousIterator>::value_type;
# else
      typedef typename std::iterator_traits<SendContiguousIterator>::value_type send_value_type;
      typedef typename std::iterator_traits<ReceiveContiguousIterator>::value_type receive_value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Sendrecv(
            YAMPI_addressof(*send_first), send_length, ::yampi::mpi_data_type_of<send_value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(),
            YAMPI_addressof(*receive_first), receive_length, ::yampi::mpi_data_type_of<receive_value_type>::value, source.mpi_rank(), receive_tag.mpi_tag(),
            communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
      int const error_code
        = MPI_Sendrecv(
            YAMPI_addressof(*send_first), send_length, ::yampi::mpi_data_type_of<send_value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(),
            YAMPI_addressof(*receive_first), receive_length, ::yampi::mpi_data_type_of<receive_value_type>::value, source.mpi_rank(), receive_tag.mpi_tag(),
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
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<SendContiguousIterator>::value_type>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ReceiveContiguousIterator>::value_type>::value,
      void>::type
    send_receive_iter(
      SendContiguousIterator const send_first, SendContiguousIterator const send_last, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
    {
      assert(send_last >= send_first && receive_last >= receive_first);
      ::yampi::send_receive_detail::send_receive_iter(
        send_first, send_last-send_first, destination, send_tag,
        receive_first, receive_last-receive_first, source, receive_tag,
        communicator, ignore_status);
    }

    template <typename SendContiguousRange, typename ReceiveContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<SendContiguousRange const>::type>::value
        and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ReceiveContiguousRange>::type>::value,
      void>::type
    send_receive_range(
      SendContiguousRange const& send_values, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveContiguousRange& receive_values, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
    {
      ::yampi::send_receive_detail::send_receive_iter(
        boost::begin(send_values), boost::end(send_values), destination, send_tag,
        boost::begin(receive_values), boost::end(receive_values), source, receive_tag,
        communicator, ignore_status);
    }

    template <typename SendContiguousRange, typename ReceiveContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<SendContiguousRange const>::type>::value
        and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ReceiveContiguousRange const>::type>::value,
      void>::type
    send_receive_range(
      SendContiguousRange const& send_values, ::yampi::rank const destination, ::yampi::tag const send_tag,
      ReceiveContiguousRange const& receive_values, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
    {
      ::yampi::send_receive_detail::send_receive_iter(
        boost::begin(send_values), boost::end(send_values), destination, send_tag,
        boost::begin(receive_values), boost::end(receive_values), source, receive_tag,
        communicator, ignore_status);
    }


    // with replacement, ignoring status
    template <typename Value>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<Value>::value,
      void>::type
    send_receive_value(
      Value& value, ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Sendrecv_replace(YAMPI_addressof(value), 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
      int const error_code
        = MPI_Sendrecv_replace(YAMPI_addressof(value), 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
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
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    send_receive_iter(
      ContiguousIterator const first, int const length,
      ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
    {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
# else
      using value_type = typename std::iterator_traits<ContiguousIterator>::value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Sendrecv_replace(YAMPI_addressof(*first), length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
      int const error_code
        = MPI_Sendrecv_replace(YAMPI_addressof(*first), length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), send_tag.mpi_tag(), source.mpi_rank(), receive_tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
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
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    send_receive_iter(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
    {
      assert(last >= first);
      ::yampi::send_receive_detail::send_receive_iter(first, last-first, destination, send_tag, source, receive_tag, communicator, ignore_status);
    }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
      void>::type
    send_receive_range(
      ContiguousRange& values,
      ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
    { ::yampi::send_receive_detail::send_receive_iter(boost::begin(values), boost::end(values), destination, send_tag, source, receive_tag, communicator, ignore_status); }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange const>::type>::value,
      void>::type
    send_receive_range(
      ContiguousRange const& values,
      ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
      ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
    { ::yampi::send_receive_detail::send_receive_iter(boost::begin(values), boost::end(values), destination, send_tag, source, receive_tag, communicator, ignore_status); }
  } // namespace send_receive_detail


  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status
  send_receive(
    SendValue const& send_value, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveValue& receive_value, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  { return ::yampi::send_receive_detail::send_receive_value(send_value, destination, send_tag, receive_value, source, receive_tag, communicator); }

  template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
      and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
    ::yampi::status>::type
  send_receive(
    SendContiguousIterator const send_first, int const send_length, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousIterator const receive_first, int const receive_length, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  { return ::yampi::send_receive_detail::send_receive_iter(send_first, send_length, destination, send_tag, receive_first, receive_length, source, receive_tag, communicator); }

  template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
      and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
    ::yampi::status>::type
  send_receive(
    SendContiguousIterator const send_first, SendContiguousIterator const send_last, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  { return ::yampi::send_receive_detail::send_receive_iter(send_first, send_last, destination, send_tag, receive_first, receive_last, source, receive_tag, communicator); }

  template <typename SendContiguousRange, typename ReceiveContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<SendContiguousRange const>::value
      and ::yampi::is_contiguous_range<ReceiveContiguousRange>::value,
    ::yampi::status>::type
  send_receive(
    SendContiguousRange const& send_values, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousRange& receive_values, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  { return ::yampi::send_receive_detail::send_receive_range(send_values, destination, send_tag, receive_values, source, receive_tag, communicator); }

  template <typename SendContiguousRange, typename ReceiveContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<SendContiguousRange const>::value
      and ::yampi::is_contiguous_range<ReceiveContiguousRange const>::value,
    ::yampi::status>::type
  send_receive(
    SendContiguousRange const& send_values, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousRange const& receive_values, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  { return ::yampi::send_receive_detail::send_receive_range(send_values, destination, send_tag, receive_values, source, receive_tag, communicator); }


  // with replacement
  template <typename Value>
  inline ::yampi::status
  send_receive(
    Value& value, ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  { return ::yampi::send_receive_detail::send_receive_value(value, destination, send_tag, source, receive_tag, communicator); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<::yampi::is_contiguous_iterator<ContiguousIterator>::value, ::yampi::status>::type
  send_receive(
    ContiguousIterator const first, int const length,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  { return ::yampi::send_receive_detail::send_receive_iter(first, length, destination, send_tag, source, receive_tag, communicator); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<::yampi::is_contiguous_iterator<ContiguousIterator>::value, ::yampi::status>::type
  send_receive(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  { return ::yampi::send_receive_detail::send_receive_iter(first, last, destination, send_tag, source, receive_tag, communicator); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<::yampi::is_contiguous_range<ContiguousRange>::value, ::yampi::status>::type
  send_receive(
    ContiguousRange& values,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  { return ::yampi::send_receive_detail::send_receive_range(values, destination, send_tag, source, receive_tag, communicator); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<::yampi::is_contiguous_range<ContiguousRange const>::value, ::yampi::status>::type
  send_receive(
    ContiguousRange const& values,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator)
  { return ::yampi::send_receive_detail::send_receive_range(values, destination, send_tag, source, receive_tag, communicator); }


  // ignoring status
  template <typename SendValue, typename ReceiveValue>
  inline void
  send_receive(
    SendValue const& send_value, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveValue& receive_value, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::send_receive_detail::send_receive_value(send_value, destination, send_tag, receive_value, source, receive_tag, communicator, ignore_status); }

  template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
      and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
    void>::type
  send_receive(
    SendContiguousIterator const send_first, int const send_length, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousIterator const receive_first, int const receive_length, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::send_receive_detail::send_receive_iter(send_first, send_length, destination, send_tag, receive_first, receive_length, source, receive_tag, communicator, ignore_status); }

  template <typename SendContiguousIterator, typename ReceiveContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<SendContiguousIterator>::value
      and ::yampi::is_contiguous_iterator<ReceiveContiguousIterator>::value,
    void>::type
  send_receive(
    SendContiguousIterator const send_first, SendContiguousIterator const send_last, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousIterator const receive_first, ReceiveContiguousIterator const receive_last, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::send_receive_detail::send_receive_iter(send_first, send_last, destination, send_tag, receive_first, receive_last, source, receive_tag, communicator, ignore_status); }

  template <typename SendContiguousRange, typename ReceiveContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<SendContiguousRange const>::value
      and ::yampi::is_contiguous_range<ReceiveContiguousRange>::value,
    void>::type
  send_receive(
    SendContiguousRange const& send_values, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousRange& receive_values, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::send_receive_detail::send_receive_range(send_values, destination, send_tag, receive_values, source, receive_tag, communicator, ignore_status); }

  template <typename SendContiguousRange, typename ReceiveContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<SendContiguousRange const>::value
      and ::yampi::is_contiguous_range<ReceiveContiguousRange const>::value,
    void>::type
  send_receive(
    SendContiguousRange const& send_values, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ReceiveContiguousRange const& receive_values, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::send_receive_detail::send_receive_range(send_values, destination, send_tag, receive_values, source, receive_tag, communicator, ignore_status); }


  // with replacement, ignoring status
  template <typename Value>
  inline void
  send_receive(
    Value& value, ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::send_receive_detail::send_receive_value(value, destination, send_tag, source, receive_tag, communicator, ignore_status); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<::yampi::is_contiguous_iterator<ContiguousIterator>::value, void>::type
  send_receive(
    ContiguousIterator const first, int const length,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::send_receive_detail::send_receive_iter(first, length, destination, send_tag, source, receive_tag, communicator, ignore_status); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<::yampi::is_contiguous_iterator<ContiguousIterator>::value, void>::type
  send_receive(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::send_receive_detail::send_receive_iter(first, last, destination, send_tag, source, receive_tag, communicator, ignore_status); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<::yampi::is_contiguous_range<ContiguousRange>::value, void>::type
  send_receive(
    ContiguousRange& values,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::send_receive_detail::send_receive_range(values, destination, send_tag, source, receive_tag, communicator, ignore_status); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<::yampi::is_contiguous_range<ContiguousRange const>::value, void>::type
  send_receive(
    ContiguousRange const& values,
    ::yampi::rank const destination, ::yampi::tag const send_tag, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::send_receive_detail::send_receive_range(values, destination, send_tag, source, receive_tag, communicator, ignore_status); }
}


# undef YAMPI_enable_if
# undef YAMPI_addressof

#endif

