#ifndef YAMPI_BLOCKING_SEND_HPP
# define YAMPI_BLOCKING_SEND_HPP

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
  namespace blocking_send_detail
  {
    template <typename Value>
    inline
    typename YAMPI_enable_if<::yampi::has_corresponding_mpi_data_type<Value>::value, void>::type
    blocking_send_value(Value const& value, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Send(YAMPI_addressof(value), 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# else
      int const error_code
        = MPI_Send(YAMPI_addressof(value), 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::blocking_send"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::blocking_send");
# endif
    }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    blocking_send_iter(ContiguousIterator const first, int const length, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
      using value_type = typename std::iterator_traits<ContiguousIterator>::value_type;
# else
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Send(YAMPI_addressof(*first), length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# else
      int const error_code
        = MPI_Send(YAMPI_addressof(*first), length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::blocking_send"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::blocking_send");
# endif
    }

    template <typename ContiguousIterator>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    blocking_send_iter(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
      assert(last >= first);
      ::yampi::blocking_send_detail::blocking_send_iter(first, last-first, destination, tag, communicator);
    }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
      void>::type
    blocking_send_range(ContiguousRange const& values, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
    { ::yampi::blocking_send_detail::blocking_send_iter(boost::begin(values), boost::end(values), destination, tag, communicator); }
  } // namespace blocking_send_detail


  template <typename Value>
  inline void
  blocking_send(Value const& value, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { ::yampi::blocking_send_detail::blocking_send_value(value, destination, tag, communicator); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<::yampi::is_contiguous_iterator<ContiguousIterator>::value, void>::type
  blocking_send(ContiguousIterator const first, int const length, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { ::yampi::blocking_send_detail::blocking_send_iter(first, length, destination, tag, communicator); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<::yampi::is_contiguous_iterator<ContiguousIterator>::value, void>::type
  blocking_send(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { ::yampi::blocking_send_detail::blocking_send_iter(first, last, destination, tag, communicator); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<::yampi::is_contiguous_range<ContiguousRange>::value, void>::type
  blocking_send(ContiguousRange const& values, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { ::yampi::blocking_send_detail::blocking_send_range(values, destination, tag, communicator); }
}


# undef YAMPI_enable_if
# undef YAMPI_addressof

#endif

