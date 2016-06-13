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
      ::yampi::has_corresponding_mpi_data_type<Value>::value,
      ::yampi::request>::type
    nonblocking_receive_value(Value& value, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto request = MPI_Request{};
#   else
      auto request = MPI_Request();
#   endif

      auto const error_code
        = MPI_Recv(YAMPI_addressof(value), 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(request));
# else
      MPI_Request request;

      int const error_code
        = MPI_Recv(YAMPI_addressof(value), 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(request));
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
      std::pair<Value, ::yampi::request> >::type
    nonblocking_receive_value(::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
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
        = MPI_Recv(YAMPI_addressof(result), 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(request));
# else
      Value result;
      MPI_Request request;

      int const error_code
        = MPI_Recv(YAMPI_addressof(result), 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(request));
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
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      ::yampi::request>::type
    nonblocking_receive_iter(ContiguousIterator const first, int const length, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
      using value_type = typename std::iterator_traits<ContiguousIterator>::value_type;
# else
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto request = MPI_Request{};
#   else
      auto request = MPI_Request();
#   endif

      auto const error_code
        = MPI_Recv(YAMPI_addressof(*first), length, ::yampi::mpi_data_type_of<value_type>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(request));
# else
      MPI_Request request;

      int const error_code
        = MPI_Recv(YAMPI_addressof(*first), length, ::yampi::mpi_data_type_of<value_type>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(request));
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
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      ::yampi::request>::type
    nonblocking_receive_iter(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
      assert(last >= first);
      return ::yampi::nonblocking_receive_detail::nonblocking_receive_iter(first, last-first, source, tag, communicator);
    }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
      ::yampi::request>::type
    nonblocking_receive_range(ContiguousRange& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
    { return ::yampi::nonblocking_receive_detail::nonblocking_receive_iter(boost::begin(values), boost::end(values), source, tag, communicator); }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange const>::type>::value,
      ::yampi::request>::type
    nonblocking_receive_range(ContiguousRange const& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
    { return ::yampi::nonblocking_receive_detail::nonblocking_receive_iter(boost::begin(values), boost::end(values), source, tag, communicator); }
  } // namespace nonblocking_receive_detail


  template <typename Value>
  inline ::yampi::request
  nonblocking_receive(Value& value, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_receive_detail::nonblocking_receive_value(value, source, tag, communicator); }

  template <typename Value>
  inline std::pair<Value, ::yampi::request>
  nonblocking_receive(::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_receive_detail::nonblocking_receive_value(source, tag, communicator); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, ::yampi::request>::type
  nonblocking_receive(ContiguousIterator const first, int const length, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_receive_detail::nonblocking_receive_iter(first, length, source, tag, communicator); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, ::yampi::request>::type
  nonblocking_receive(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_receive_detail::nonblocking_receive_iter(first, last, source, tag, communicator); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value,
    ::yampi::request>::type
  nonblocking_receive(ContiguousRange& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_receive_detail::nonblocking_receive_range(values, source, tag, communicator); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange const>::value,
    ::yampi::request>::type
  nonblocking_receive(ContiguousRange const& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_receive_detail::nonblocking_receive_range(values, source, tag, communicator); }
}


# undef YAMPI_enable_if
# undef YAMPI_addressof

#endif

