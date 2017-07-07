#ifndef YAMPI_BLOCKING_SEND_HPP
# define YAMPI_BLOCKING_SEND_HPP

# include <boost/config.hpp>

/*
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
# include <yampi/error.hpp>

/*
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
*/


namespace yampi
{
  template <typename Value>
  inline void blocking_send(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag)
  {
    int const error_code
      = MPI_Send(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_send", environment);
  }
  /*
  namespace blocking_send_detail
  {
    template <typename Value>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<Value>::value,
      void>::type
    blocking_send(
      Value const& value,
      ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator const communicator)
    {
      int const error_code
        = MPI_Send(
            const_cast<Value*>(YAMPI_addressof(value)), 1,
            ::yampi::datatype_of<Value>::call().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::blocking_send");
    }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<
        typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    blocking_send(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator const communicator)
    {
      assert(last >= first);

      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
      int const error_code
        = MPI_Send(
            YAMPI_addressof(*first), last-first,
            ::yampi::datatype_of<value_type>::call().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::blocking_send");
    }
  } // namespace blocking_send_detail


  template <typename Value>
  inline
  typename YAMPI_enable_if<
    not ::yampi::is_contiguous_range<Value const>::value,
    void>::type
  blocking_send(
    Value const& value,
    ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator const communicator)
  {
    ::yampi::blocking_send_detail::blocking_send(
      value, destination, tag, communicator);
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
    void>::type
  blocking_send(
    ContiguousIterator const first, ContiguousIterator const last,
    ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator const communicator)
  {
    ::yampi::blocking_send_detail::blocking_send(
      first, last, destination, tag, communicator);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value,
    void>::type
  blocking_send(
    ContiguousRange const& values,
    ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator const communicator)
  {
    ::yampi::blocking_send_detail::blocking_send(
      boost::begin(values), boost::end(values), destination, tag, communicator);
  }
  */
}


/*
# undef YAMPI_enable_if
# undef YAMPI_addressof
*/

#endif

