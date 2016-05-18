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
  inline
  typename YAMPI_enable_if<
    ::yampi::has_corresponding_mpi_data_type<Value>::value,
    ::yampi::request<Value> >::type
  nonblocking_receive(Value& value, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto request = MPI_Request{};
#   else
    auto request = MPI_Request();
#   endif

    auto const error_code
      = MPI_Recv(&value, 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &request);
# else
    MPI_Request request;

    int const error_code
      = MPI_Recv(&value, 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &request);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::nonblocking_receive"};

    return ::yampi::request{request};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::nonblocking_receive");

    return ::yampi::request(request);
# endif
  }

  template <typename Value>
  inline
  typename YAMPI_enable_if<
    ::yampi::has_corresponding_mpi_data_type<Value>::value,
    std::pair<Value, ::yampi::request<Value> > >::type
  nonblocking_receive(::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto result = Value{};
    auto request = MPI_Request{};
#   else
    auto result = Value();
    auto request = MPI_Request();
#   endif

    auto const error_code
      = MPI_Recv(&result, 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &request);
# else
    Value result;
    MPI_Request request;

    int const error_code
      = MPI_Recv(&result, 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &request);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::nonblocking_receive"};

    return std::make_pair(result, ::yampi::request{request});
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::nonblocking_receive");

    return std::make_pair(result, ::yampi::request(request));
# endif
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    ::yampi::request<Value> >::type
  nonblocking_receive(ContiguousIterator const first, int const length, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  {
    typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto request = MPI_Request{};
#   else
    auto request = MPI_Request();
#   endif

    auto const error_code
      = MPI_Recv(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &request);
# else
    MPI_Request request;

    int const error_code
      = MPI_Recv(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &request);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::nonblocking_receive"};

    return ::yampi::request{request};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::nonblocking_receive");

    return ::yampi::request(request);
# endif
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    ::yampi::request<Value> >::type
  nonblocking_receive(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  {
    assert(last >= first);
    return ::yampi::nonblocking_receive(first, last-first, source, tag, communicator);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value
      and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
    ::yampi::request<Value> >::type
  nonblocking_receive(ContiguousRange& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_receive(boost::begin(values), boost::end(values), source, tag, communicator); }
}


# undef YAMPI_enable_if

#endif

