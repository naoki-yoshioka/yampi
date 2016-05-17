#ifndef YAMPI_BLOCKING_RECEIVE_HPP
# define YAMPI_BLOCKING_RECEIVE_HPP

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
# include <yampi/environment.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/status.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_enable_if boost::enable_if
# endif


namespace yampi
{
  // Blocking receive
  template <typename Value>
  inline
  typename YAMPI_enable_if<
    ::yampi::has_corresponding_mpi_data_type<Value>::value,
    ::yampi::status<Value> >::type
  blocking_receive(Value& value, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment&)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto stat = MPI_Status{};
#   else
    auto stat = MPI_Status();
#   endif

    auto const error_code
      = MPI_Recv(&value, 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &stat);
# else
    MPI_Status stat;

    int const error_code
      = MPI_Recv(&value, 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &stat);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::blocking_receive"};

    return ::yampi::status{stat};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_receive");

    return ::yampi::status(stat);
# endif
  }

  template <typename Value>
  inline
  typename YAMPI_enable_if<
    ::yampi::has_corresponding_mpi_data_type<Value>::value,
    std::pair<Value, ::yampi::status<Value> > >::type
  blocking_receive(::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment&)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto result = Value{};
    auto stat = MPI_Status{};
#   else
    auto result = Value();
    auto stat = MPI_Status();
#   endif

    auto const error_code
      = MPI_Recv(&result, 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &stat);
# else
    Value result;
    MPI_Status stat;

    int const error_code
      = MPI_Recv(&result, 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &stat);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::blocking_receive"};

    return std::make_pair(result, ::yampi::status{stat});
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_receive");

    return std::make_pair(result, ::yampi::status(stat));
# endif
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    ::yampi::status<Value> >::type
  blocking_receive(ContiguousIterator const first, int const length, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment&)
  {
    typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto stat = MPI_Status{};
#   else
    auto stat = MPI_Status();
#   endif

    auto const error_code
      = MPI_Recv(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &stat);
# else
    MPI_Status stat;

    int const error_code
      = MPI_Recv(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), &stat);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::blocking_receive"};

    return ::yampi::status{stat};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_receive");

    return ::yampi::status(stat);
# endif
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    ::yampi::status<Value> >::type
  blocking_receive(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment& env)
  {
    assert(last >= first);
    ::yampi::blocking_receive(first, last-first, source, tag, communicator, env);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value
      and ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
    ::yampi::status<Value> >::type
  blocking_receive(ContiguousRange& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment& env)
  { ::yampi::blocking_receive(boost::begin(values), boost::end(values), source, tag, communicator, env); }


  // Blocking receive (ignoring status)
  template <typename Value>
  inline
  typename YAMPI_enable_if<::yampi::has_corresponding_mpi_data_type<Value>::value, void>::type
  blocking_receive(Value& value, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment&, ::yampi::ignore_status_t const)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
    auto const error_code
      = MPI_Recv(&value, 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
    int const error_code
      = MPI_Recv(&value, 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::blocking_receive"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_receive");
# endif
  }

  template <typename Value>
  inline
  typename YAMPI_enable_if<::yampi::has_corresponding_mpi_data_type<Value>::value, Value>::type
  blocking_receive(::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment&, ::yampi::ignore_status_t const)
  {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    auto result = Value{};
#   else
    auto result = Value();
#   endif

    auto const error_code
      = MPI_Recv(&result, 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
    Value result;

    int const error_code
      = MPI_Recv(&result, 1, ::yampi::mpi_data_type_of<Value>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::blocking_receive"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_receive");
# endif

    return result;
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    void>::type
  blocking_receive(ContiguousIterator const first, int const length, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment&, ::yampi::ignore_status_t const)
  {
    typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
    auto const error_code
      = MPI_Recv(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
    int const error_code
      = MPI_Recv(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::blocking_receive"};
# else
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_receive");
# endif
  }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_iterator<ContiguousIterator>::value
      and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
    void>::type
  blocking_receive(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment&, ::yampi::ignore_status_t const ignore)
  {
    assert(last >= first);
    ::yampi::blocking_receive(first, last-first, source, tag, communicator, env, ignore);
  }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value
      and ::yampi::has_corresponding_mpi_data_type<typename boost::range_iterator<ContiguousRange>::type>::value,
    void>::type
  blocking_receive(ContiguousRange& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::environment&, ::yampi::ignore_status_t const ignore)
  { ::yampi::blocking_receive(boost::begin(values), boost::end(values), source, tag, communicator, env, ignore); }
}


# undef YAMPI_enable_if

#endif

