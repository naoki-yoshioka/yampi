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
  class broadcast
  {
    ::yampi::rank root_;
    ::yampi::communicator comm_;

   public:
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    BOOST_CONSTEXPR broadcast() BOOST_NOEXCEPT_OR_NOTHROW
      : root_{0}, comm_{::yampi::world}
    { }

    BOOST_CONSTEXPR broadcast(::yampi::rank const root) BOOST_NOEXCEPT_OR_NOTHROW
      : root_{root}, comm_{::yampi::world}
    { }

    BOOST_CONSTEXPR broadcast(::yampi::communicator const comm) BOOST_NOEXCEPT_OR_NOTHROW
      : root_{0}, comm_{comm}
    { }

    BOOST_CONSTEXPR broadcast(::yampi::rank const root, ::yampi::communicator const comm) BOOST_NOEXCEPT_OR_NOTHROW
      : root_{root}, comm_{comm}
    { }
# else
    BOOST_CONSTEXPR broadcast() BOOST_NOEXCEPT_OR_NOTHROW
      : root_(0), comm_(::yampi::world)
    { }

    BOOST_CONSTEXPR broadcast(::yampi::rank const root) BOOST_NOEXCEPT_OR_NOTHROW
      : root_(root), comm_(::yampi::world)
    { }

    BOOST_CONSTEXPR broadcast(::yampi::communicator const comm) BOOST_NOEXCEPT_OR_NOTHROW
      : root_(0), comm_(comm)
    { }

    BOOST_CONSTEXPR broadcast(::yampi::rank const root, ::yampi::communicator const comm) BOOST_NOEXCEPT_OR_NOTHROW
      : root_(root), comm_(comm)
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
    call(Value& value, ::yampi::environment&) const
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
    call(ContiguousIterator const first, int const length, ::yampi::environment&) const
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
    call(ContiguousIterator const first, ContiguousIterator const last, ::yampi::environment& env) const
    {
      assert(last >= first);
      call(first, last-first, env);
    }

    template <typename ContiguousRange>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange>::value
        and ::yampi::has_corresponding_mpi_data_type<typename std::range_value<ContiguousRange>::type>::value,
      void>::type
    call(ContiguousRange& values, ::yampi::environment& env) const
    { call(boost::begin(values), boost::end(values), env); }
  };
}


# undef YAMPI_enable_if

#endif

