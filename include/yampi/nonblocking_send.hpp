#ifndef YAMPI_NONBLOCKING_SEND_HPP
# define YAMPI_NONBLOCKING_SEND_HPP

# include <boost/config.hpp>

/*
# include <cassert>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif
# include <vector>
# include <iterator>
*/
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

/*
# include <yampi/has_corresponding_datatype.hpp>
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
# include <yampi/request.hpp>
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
  template <typename Value>
  inline ::yampi::request nonblocking_send(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const destination, ::yampi::tag const tag)
  {
    MPI_Request request;
    int const error_code
      = MPI_Isend(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          YAMPI_addressof(request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::nonblocking_send", environment);

    return ::yampi::request(request);
  }
  /*
  namespace nonblocking_send_detail
  {
    template <typename Value>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<Value>::value,
      ::yampi::request>::type
    nonblocking_send(
      Value const& value,
      ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator const communicator)
    {
      MPI_Request request;
      int const error_code
        = MPI_Isend(
            const_cast<Value*>(YAMPI_addressof(value)), 1,
            ::yampi::datatype_of<Value>::call().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            YAMPI_addressof(request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::nonblocking_send");

      return ::yampi::request(request);
    }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<
        typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      ::yampi::request>::type
    nonblocking_send(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator const communicator)
    {
      assert(last >= first);

      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
      MPI_Request request;
      int const error_code
        = MPI_Isend(
            YAMPI_addressof(*first), last-first,
            ::yampi::datatype_of<value_type>::call().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            YAMPI_addressof(request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::nonblocking_send");

      return ::yampi::request(request);
    }
  } // namespace nonblocking_send_detail


  template <typename Value>
  inline
  typename YAMPI_enable_if<
    not ::yampi::is_contiguous_range<Value const>::value,
    ::yampi::request>::type
  nonblocking_send(
    Value const& value,
    ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::nonblocking_send_detail::nonblocking_send(
      value, destination, tag, communicator);
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
    ::yampi::request>::type
  nonblocking_send(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::nonblocking_send_detail::nonblocking_send(
      first, last, destination, tag, communicator);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange const>::value,
    ::yampi::request>::type
  nonblocking_send(
    ContiguousRange const& values,
    ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator const communicator)
  {
    return ::yampi::nonblocking_send_detail::nonblocking_send(
      boost::begin(values), boost::end(values), destination, tag, communicator);
  }
  */
}


/*
# undef YAMPI_enable_if
*/
# undef YAMPI_addressof

#endif

