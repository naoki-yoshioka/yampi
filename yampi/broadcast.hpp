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
  class broadcast
  {
    ::yampi::communicator comm_;
    ::yampi::rank root_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    broadcast() = delete;
# else
   public:
    broadcast();

   private:
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    explicit BOOST_CONSTEXPR broadcast(::yampi::communicator const comm, ::yampi::rank const root = ::yampi::rank{0}) BOOST_NOEXCEPT_OR_NOTHROW
      : comm_{comm}, root_{root}
    { }
# else
    explicit BOOST_CONSTEXPR broadcast(::yampi::communicator const comm, ::yampi::rank const root = ::yampi::rank(0)) BOOST_NOEXCEPT_OR_NOTHROW
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
    typename YAMPI_enable_if<not ::yampi::is_contiguous_range<Value>::value, void>::type
    call(Value& value) const
    { do_call_value(value); }

    template <typename ContiguousRange>
    typename YAMPI_enable_if<::yampi::is_contiguous_range<ContiguousRange>::value, void>::type
    call(ContiguousRange& values) const
    { do_call_range(values); }

    template <typename ContiguousRange>
    typename YAMPI_enable_if<::yampi::is_contiguous_range<ContiguousRange const>::value, void>::type
    call(ContiguousRange const& values) const
    { do_call_range(values); }


    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      void>::type
    call(ContiguousIterator const first, int const length) const
    { do_call_iter(first, length); }

    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      void>::type
    call(ContiguousIterator const first, ContiguousIterator const last) const
    { do_call_iter(first, last); }


   private:
    template <typename Value>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<Value>::value,
      void>::type
    do_call_value(Value& value) const
    {
# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Bcast(YAMPI_addressof(value), 1, ::yampi::mpi_data_type_of<Value>::value, root_.mpi_rank(), comm_.mpi_comm());
# else
      int const error_code
        = MPI_Bcast(YAMPI_addressof(value), 1, ::yampi::mpi_data_type_of<Value>::value, root_.mpi_rank(), comm_.mpi_comm());
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
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    do_call_iter(ContiguousIterator const first, int const length) const
    {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
      using value_type = typename std::iterator_traits<ContiguousIterator>::value_type;
# else
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Bcast(YAMPI_addressof(*first), length, ::yampi::mpi_data_type_of<value_type>::value, root_.mpi_rank(), comm_.mpi_comm());
# else
      int const error_code
        = MPI_Bcast(YAMPI_addressof(*first), length, ::yampi::mpi_data_type_of<value_type>::value, root_.mpi_rank(), comm_.mpi_comm());
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
      ::yampi::has_corresponding_mpi_data_type<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    do_call_iter(ContiguousIterator const first, ContiguousIterator const last) const
    {
      assert(last >= first);
      do_call_iter(first, last-first);
    }

    template <typename ContiguousRange>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange>::type>::value,
      void>::type
    do_call_range(ContiguousRange& values) const
    {
      using boost::begin;
      using boost::end;
      do_call_iter(begin(values), end(values));
    }

    template <typename ContiguousRange>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_mpi_data_type<typename boost::range_value<ContiguousRange const>::type>::value,
      void>::type
    do_call_range(ContiguousRange const& values) const
    {
      using boost::begin;
      using boost::end;
      do_call_iter(begin(values), end(values));
    }
  };
}


# undef YAMPI_enable_if
# undef YAMPI_addressof

#endif

