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

# include <yampi/has_corresponding_datatype.hpp>
# include <yampi/is_contiguous_iterator.hpp>
# include <yampi/is_contiguous_range.hpp>
# include <yampi/datatype_of.hpp>
# include <yampi/datatype.hpp>
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
   private:
    broadcast();

   public:
# endif

    explicit BOOST_CONSTEXPR broadcast(
      ::yampi::communicator const comm,
      ::yampi::rank const root = ::yampi::rank(0)) BOOST_NOEXCEPT_OR_NOTHROW
      : comm_(comm), root_(root)
    { }

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
      not ::yampi::is_contiguous_range<Value>::value,
      void>::type
    call(Value& value) const
    { do_call(value); }

    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      void>::type
    call(ContiguousIterator const first, ContiguousIterator const last) const
    { do_call(first, last); }

    template <typename ContiguousRange>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange>::value,
      void>::type
    call(ContiguousRange& values) const
    { do_call(boost::begin(values), boost::end(values)); }

    template <typename ContiguousRange>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange>::value,
      void>::type
    call(ContiguousRange const& values) const
    { do_call(boost::begin(values), boost::end(values)); }


   private:
    template <typename Value>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<Value>::value,
      void>::type
    do_call(Value& value) const
    {
      int const error_code
        = MPI_Bcast(
            YAMPI_addressof(value), 1, ::yampi::datatype_of<Value>::call().mpi_datatype(),
            root_.mpi_rank(), comm_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call");
    }

    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<
        typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    do_call(
      ContiguousIterator const first, ContiguousIterator const last) const
    {
      assert(last >= first);
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
      int const error_code
        = MPI_Bcast(
            YAMPI_addressof(*first), last-first,
            ::yampi::datatype_of<value_type>::call().mpi_datatype(),
            root_.mpi_rank(), comm_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call");
    }
  };
}


# undef YAMPI_enable_if
# undef YAMPI_addressof

#endif

