#ifndef YAMPI_NONBLOCKING_RECEIVE_HPP
# define YAMPI_NONBLOCKING_RECEIVE_HPP

# include <boost/config.hpp>

# include <cassert>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif
# include <iterator>
# include <utility>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/value_type.hpp>

# include <mpi.h>

# include <yampi/has_corresponding_datatype.hpp>
# include <yampi/is_contiguous_iterator.hpp>
# include <yampi/is_contiguous_range.hpp>
# include <yampi/datatype_of.hpp>
# include <yampi/datatype.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/request.hpp>
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
  namespace nonblocking_receive_detail
  {
    template <typename Value>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<Value>::value,
      ::yampi::request>::type
    nonblocking_receive(
      Value& value,
      ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator const communicator)
    {
      MPI_Request request;
      int const error_code
        = MPI_Recv(
            YAMPI_addressof(value), 1, ::yampi::datatype_of<Value>::call().mpi_datatype(),
            source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::nonblocking_receive");

      return ::yampi::request(request);
    }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<
        typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      ::yampi::request>::type
    nonblocking_receive(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator const communicator)
    {
      assert(last >= first);

      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
      MPI_Request request;
      int const error_code
        = MPI_Recv(
            YAMPI_addressof(*first), last-first,
            ::yampi::datatype_of<value_type>::call().mpi_datatype(),
            source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::nonblocking_receive");

      return ::yampi::request(request);
    }
  } // namespace nonblocking_receive_detail


  template <typename Value>
  inline
  typename YAMPI_enable_if<
    not ::yampi::is_contiguous_range<Value>::value,
    ::yampi::request>::type
  nonblocking_receive(
    Value& value,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::nonblocking_receive_detail::nonblocking_receive(
      value, source, tag, communicator);
  }

  template <typename Value>
  inline std::pair<Value, ::yampi::request>
  nonblocking_receive(
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator const communicator)
  {
    Value value;
    ::yampi::request request
      = ::yampi::nonblocking_receive_detail::nonblocking_receive(
          value, source, tag, communicator);
    return std::make_pair(value, request);
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
    ::yampi::request>::type
  nonblocking_receive(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::nonblocking_receive_detail::nonblocking_receive(
      first, last, source, tag, communicator);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value,
    ::yampi::request>::type
  nonblocking_receive(
    ContiguousRange& values,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::nonblocking_receive_detail::nonblocking_receive_range(
      boost::begin(values), boost::end(values), source, tag, communicator);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange const>::value,
    ::yampi::request>::type
  nonblocking_receive(
    ContiguousRange const& values,
    ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::nonblocking_receive_detail::nonblocking_receive_range(
      boost::begin(values), boost::end(values), source, tag, communicator);
  }
}


# undef YAMPI_enable_if
# undef YAMPI_addressof

#endif

