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

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/value_type.hpp>

# include <mpi.h>

# include <yampi/has_corresponding_mpi_data_type.hpp>
# include <yampi/is_contiguous_iterator.hpp>
# include <yampi/is_contiguous_range.hpp>
# include <yampi/mpi_data_type_of.hpp>
# include <yampi/environment.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
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
  typename YAMPI_enable_if<::yampi::has_corresponding_mpi_data_type<Value>::value, void>::type
  blocking_send(Value const& value, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment&)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
    auto const error_code
      = MPI_Send(&value, 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# else
    int const error_code
      = MPI_Send(&value, 1, ::yampi::mpi_data_type_of<Value>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
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
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    void>::type
  blocking_send(ContiguousIterator const first, int const length, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment&)
  {
    typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
    auto const error_code
      = MPI_Send(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# else
    int const error_code
      = MPI_Send(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
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
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    void>::type
  blocking_send(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment& env)
  {
    assert(last >= first);
    ::yampi::blocking_send(first, last-first, destination, tag, communicator, env);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value
      and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
    void>::type
  blocking_send(ContiguousRange const& values, ::yampi::rank const destination, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment& env)
  { ::yampi::blocking_send(boost::begin(values), boost::end(values), destination, tag, communicator, env); }
}


# undef YAMPI_enable_if

#endif

