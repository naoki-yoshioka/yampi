#ifndef YAMPI_BROADCAST_HPP
# define YAMPI_BROADCAST_HPP

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
  class broadcast
  {
    ::yampi::communicator comm_;
    ::yampi::rank root_;

   public:
    BOOST_DELETED_FUNCTION(broadcast())

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    explicit BOOST_CONSTEXPR broadcast(::yampi::communicator const comm, ::yampi::rank const root = ::yampi::rank{0}) BOOST_NOEXCEPT_OR_NOTHROW
      : comm_{comm}, root_{root}
    { }
# else
    explicit BOOST_CONSTEXPR broadcast(::yampi::communicator const comm, ::yampi::rank const root = ::yampi::rank{0}) BOOST_NOEXCEPT_OR_NOTHROW
      : comm_(comm), root_(root)
    { }
# endif

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    broadcast(broadcast const&) = default;
    broadcast& operator=(broadcast const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    broadcast(broadcast&&) = default;
    broadcast& operator=(broadcast&&) = default;
#   endif
    ~broadcast() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif


    template <typename Value>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<Value>::value,
      void>::type
    call(Value& value) const
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Bcast(&value, 1, ::yampi::mpi_data_type_of<Value>::value, root_.mpi_rank(), comm_.mpi_comm());
# else
      int const error_code
        = MPI_Bcast(&value, 1, ::yampi::mpi_data_type_of<Value>::value, root_.mpi_rank(), comm_.mpi_comm());
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::broadcast::call"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call");
# endif
    }

    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    call(ContiguousIterator const first, int const length) const
    {
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Bcast(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, root_.mpi_rank(), comm_.mpi_comm());
# else
      int const error_code
        = MPI_Bcast(&*first, length, ::yampi::mpi_data_type_of<value_type>::value, root_.mpi_rank(), comm_.mpi_comm());
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::broadcast::call"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call");
# endif
    }

    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    call(ContiguousIterator const first, ContiguousIterator const last) const
    {
      assert(last >= first);
      call(first, last-first);
    }

    template <typename ContiguousRange>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::range_value<ContiguousRange>::type>::value,
      void>::type
    call(ContiguousRange& values) const
    { call(boost::begin(values), boost::end(values)); }
  };
}


# undef YAMPI_enable_if

#endif

