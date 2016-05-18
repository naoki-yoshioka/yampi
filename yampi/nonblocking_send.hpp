#ifndef YAMPI_NONBLOCKING_SEND_HPP
# define YAMPI_NONBLOCKING_SEND_HPP

# include <boost/config.hpp>

# include <cassert>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif
# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# endif
# include <vector>
# include <iterator>

# include <boost/array.hpp>

# include <mpi.h>

# include <yampi/has_corresponding_mpi_data_type.hpp>
# include <yampi/mpi_data_type_of.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/request.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_enable_if boost::enable_if
# endif


namespace yampi
{
  template <typename Value>
  typename YAMPI_enable_if<::yampi::has_corresponding_mpi_data_type<Value>::value, ::yampi::request>::type
  nonblocking_send(Value const& value, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto request = MPI_Request{};
#   else
    auto request = MPI_Request();
#   endif

    auto const error_code
      = MPI_Isend(&value, 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &request);
# else
    MPI_Request request;

    int const error_code
      = MPI_Isend(&value, 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &request);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::nonblocking_send"};

    return ::yampi::request{request};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::nonblocking_send");

    return ::yampi::request(request);
# endif
  }

  template <typename ContiguousIterator>
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    ::yampi::request>::type
  nonblocking_send(ContiguousIterator const first, int const length, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator
  {
    typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto request = MPI_Request{};
#   else
    auto request = MPI_Request();
#   endif

    auto const error_code
      = MPI_Isend(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &request);
# else
    MPI_Request request;

    int const error_code
      = MPI_Isend(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &request);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::nonblocking_send"};

    return ::yampi::request{request};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::nonblocking_send");

    return ::yampi::request(request);
# endif
  }

  template <typename ContiguousIterator>
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    ::yampi::request>::type
  nonblocking_send(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
  {
    assert(last >= first);
    return ::yampi::nonblocking_send(first, last-first, destination, tag, communicator);
  }

  template <typename ContiguousRange>
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value
      and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
    ::yampi::request>::type
  nonblocking_send(ContiguousRange& values, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_send(boost::begin(values), boost::end(values), destination, tag, communicator); }
}


# undef YAMPI_enable_if

#endif

