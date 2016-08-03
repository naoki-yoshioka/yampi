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
# include <yampi/status.hpp>
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
  namespace blocking_receive_detail
  {
    // Blocking receive
    template <typename Value>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<Value>::value,
      ::yampi::status>::type
    blocking_receive_value(Value& value, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto stat = MPI_Status{};
#   else
      auto stat = MPI_Status();
#   endif

      auto const error_code
        = MPI_Recv(YAMPI_addressof(value), 1, ::yampi::datatype_of<Value>::call().mpi_datatype(), source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(stat));
# else
      MPI_Status stat;

      int const error_code
        = MPI_Recv(YAMPI_addressof(value), 1, ::yampi::datatype_of<Value>::call().mpi_datatype(), source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(stat));
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
      ::yampi::has_corresponding_datatype<Value>::value,
      std::pair<Value, ::yampi::status> >::type
    blocking_receive_value(::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
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
        = MPI_Recv(YAMPI_addressof(result), 1, ::yampi::datatype_of<Value>::call().mpi_datatype(), source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(stat));
# else
      Value result;
      MPI_Status stat;

      int const error_code
        = MPI_Recv(YAMPI_addressof(result), 1, ::yampi::datatype_of<Value>::call().mpi_datatype(), source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(stat));
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
      ::yampi::has_corresponding_datatype<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      ::yampi::status>::type
    blocking_receive_iter(ContiguousIterator const first, int const length, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
      using value_type = typename std::iterator_traits<ContiguousIterator>::value_type;
# else
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto stat = MPI_Status{};
#   else
      auto stat = MPI_Status();
#   endif

      auto const error_code
        = MPI_Recv(YAMPI_addressof(*first), length, ::yampi::datatype_of<value_type>::call().mpi_datatype(), source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(stat));
# else
      MPI_Status stat;

      int const error_code
        = MPI_Recv(YAMPI_addressof(*first), length, ::yampi::datatype_of<value_type>::call().mpi_datatype(), source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(stat));
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
      ::yampi::has_corresponding_datatype<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      ::yampi::status>::type
    blocking_receive_iter(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
      assert(last >= first);
      return ::yampi::blocking_receive_detail::blocking_receive_iter(first, last-first, source, tag, communicator);
    }


    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<typename boost::range_value<ContiguousRange>::type>::value,
      ::yampi::status>::type
    blocking_receive_range(ContiguousRange& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
      return ::yampi::blocking_receive_detail::blocking_receive_iter(
        boost::begin(values), boost::end(values), source, tag, communicator);
    }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<typename boost::range_value<ContiguousRange const>::type>::value,
      ::yampi::status>::type
    blocking_receive_range(ContiguousRange const& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
    {
      return ::yampi::blocking_receive_detail::blocking_receive_iter(
        boost::begin(values), boost::end(values), source, tag, communicator);
    }


    // Blocking receive (ignoring status)
    template <typename Value>
    inline
    typename YAMPI_enable_if< ::yampi::has_corresponding_datatype<Value>::value, void>::type
    blocking_receive_value(Value& value, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Recv(YAMPI_addressof(value), 1, ::yampi::datatype_of<Value>::call().mpi_datatype(), source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
      int const error_code
        = MPI_Recv(YAMPI_addressof(value), 1, ::yampi::datatype_of<Value>::call().mpi_datatype(), source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
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
    typename YAMPI_enable_if< ::yampi::has_corresponding_datatype<Value>::value, Value>::type
    blocking_receive_value(::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto result = Value{};
#   else
      auto result = Value();
#   endif

      auto const error_code
        = MPI_Recv(YAMPI_addressof(result), 1, ::yampi::datatype_of<Value>::call().mpi_datatype(), source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
      Value result;

      int const error_code
        = MPI_Recv(YAMPI_addressof(result), 1, ::yampi::datatype_of<Value>::call().mpi_datatype(), source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
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
      ::yampi::has_corresponding_datatype<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    blocking_receive_iter(ContiguousIterator const first, int const length, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const)
    {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
      using value_type = typename std::iterator_traits<ContiguousIterator>::value_type;
# else
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Recv(YAMPI_addressof(*first), length, ::yampi::datatype_of<value_type>::call().mpi_datatype(), source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else
      int const error_code
        = MPI_Recv(YAMPI_addressof(*first), length, ::yampi::datatype_of<value_type>::call().mpi_datatype(), source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
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
      ::yampi::has_corresponding_datatype<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    blocking_receive_iter(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore)
    {
      assert(last >= first);
      ::yampi::blocking_receive_detail::blocking_receive_iter(first, last-first, source, tag, communicator, ignore);
    }


    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<typename boost::range_iterator<ContiguousRange>::type>::value,
      void>::type
    blocking_receive_range(ContiguousRange& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore)
    { ::yampi::blocking_receive_detail::blocking_receive_iter(boost::begin(values), boost::end(values), source, tag, communicator, ignore); }

    template <typename ContiguousRange>
    inline
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<typename boost::range_iterator<ContiguousRange const>::type>::value,
      void>::type
    blocking_receive_range(ContiguousRange const& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore)
    { ::yampi::blocking_receive_detail::blocking_receive_iter(boost::begin(values), boost::end(values), source, tag, communicator, ignore); }
  } // namespace blocking_receive_detail


  // Blocking receive
  template <typename Value>
  inline yampi::status
  blocking_receive(Value& value, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::blocking_receive_detail::blocking_receive_value(value, source, tag, communicator); }

  template <typename Value>
  inline std::pair<Value, ::yampi::status>
  blocking_receive(::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::blocking_receive_detail::blocking_receive_value<Value>(source, tag, communicator); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, ::yampi::status>::type
  blocking_receive(ContiguousIterator const first, int const length, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::blocking_receive_detail::blocking_receive_iter(first, length, source, tag, communicator); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, ::yampi::status>::type
  blocking_receive(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::blocking_receive_detail::blocking_receive_iter(first, last, source, tag, communicator); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange>::value, ::yampi::status>::type
  blocking_receive(ContiguousRange& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::blocking_receive_detail::blocking_receive_range(values, source, tag, communicator); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange const>::value, ::yampi::status>::type
  blocking_receive(ContiguousRange const& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator)
  { return ::yampi::blocking_receive_detail::blocking_receive_range(values, source, tag, communicator); }


  // Blocking receive (ignoring status)
  template <typename Value>
  inline void
  blocking_receive(Value& value, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::blocking_receive_detail::blocking_receive_value(value, source, tag, communicator, ignore_status); }

  template <typename Value>
  inline Value
  blocking_receive(::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { return ::yampi::blocking_receive_detail::blocking_receive_value<Value>(source, tag, communicator, ignore_status); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, void>::type
  blocking_receive(ContiguousIterator const first, int const length, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::blocking_receive_detail::blocking_receive_iter(first, length, source, tag, communicator, ignore_status); }

  template <typename ContiguousIterator>
  inline
  typename YAMPI_enable_if< ::yampi::is_contiguous_iterator<ContiguousIterator>::value, void>::type
  blocking_receive(ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::blocking_receive_detail::blocking_receive_iter(first, last, source, tag, communicator, ignore_status); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange>::value,
    void>::type
  blocking_receive(ContiguousRange& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::blocking_receive_detail::blocking_receive_range(values, source, tag, communicator, ignore_status); }

  template <typename ContiguousRange>
  inline
  typename YAMPI_enable_if<
    ::yampi::is_contiguous_range<ContiguousRange const>::value,
    void>::type
  blocking_receive(ContiguousRange const& values, ::yampi::rank const source, ::yampi::tag const tag, ::yampi::communicator const communicator, ::yampi::ignore_status_t const ignore_status)
  { ::yampi::blocking_receive_detail::blocking_receive_range(values, source, tag, communicator, ignore_status); }
}


# undef YAMPI_enable_if
# undef YAMPI_addressof

#endif

