#ifndef YAMPI_NONBLOCKING_SEND_HPP
# define YAMPI_NONBLOCKING_SEND_HPP

# include <boost/config.hpp>

# include <cassert>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif
# include <vector>
# include <iterator>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

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
#   define YAMPI_enable_if boost::enable_if_c
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  namespace nonblocking_send_detail
  {
    template <typename Value>
    inline
    typename YAMPI_enable_if< ::yampi::has_corresponding_mpi_data_type<Value>::value, ::yampi::request>::type
    nonblocking_send_value(Value const& value, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto request = MPI_Request{};
#   else
      auto request = MPI_Request();
#   endif

      auto const error_code
        = MPI_Isend(const_cast<Value*>(YAMPI_addressof(value)), 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(request));
# else
      MPI_Request request;

      int const error_code
        = MPI_Isend(const_cast<Value*>(YAMPI_addressof(value)), 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(request));
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
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      ::yampi::request>::type
    nonblocking_send_iter(ContiguousIterator const first, int const length, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
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
        = MPI_Isend(YAMPI_addressof(*first), length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(request));
# else
      MPI_Request request;

      int const error_code
        = MPI_Isend(YAMPI_addressof(*first), length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(request));
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
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      ::yampi::request>::type
    nonblocking_send_iter(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
      assert(last >= first);
      return ::yampi::nonblocking_send_detail::nonblocking_send_iter(first, last-first, destination, tag, communicator);
    }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
      ::yampi::request>::type
    nonblocking_send_range(ContiguousRange& values, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
    { return ::yampi::nonblocking_send_detail::nonblocking_send_iter(boost::begin(values), boost::end(values), destination, tag, communicator); }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange const>::type>::value,
      ::yampi::request>::type
    nonblocking_send_range(ContiguousRange const& values, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
    { return ::yampi::nonblocking_send_detail::nonblocking_send_iter(boost::begin(values), boost::end(values), destination, tag, communicator); }
  } // namespace nonblocking_send_detail


  template <typename Value>
  inline ::yampi::request
  nonblocking_send(Value const& value, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_send_detail::nonblocking_send_value(value, destination, tag, communicator); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, ::yampi::request>::type
  nonblocking_send(ContiguousIterator const first, int const length, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_send_detail::nonblocking_send_iter(first, length, destination, tag, communicator); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, ::yampi::request>::type
  nonblocking_send(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_send_detail::nonblocking_send_iter(first, last, destination, tag, communicator); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange>::value, ::yampi::request>::type
  nonblocking_send(ContiguousRange& values, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_send_detail::nonblocking_send_range(values, destination, tag, communicator); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange const>::value, ::yampi::request>::type
  nonblocking_send(ContiguousRange const& values, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::nonblocking_send_detail::nonblocking_send_range(values, destination, tag, communicator); }
}


# undef YAMPI_enable_if
# undef YAMPI_addressof

#endif

