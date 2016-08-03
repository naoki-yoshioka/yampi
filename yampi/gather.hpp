#ifndef YAMPI_GATHER_HPP
# define YAMPI_GATHER_HPP

# include <boost/config.hpp>

# include <cassert>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
#   include <boost/type_traits/is_same.hpp>
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
# include <yampi/nonroot_call_on_root_error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
#   define YAMPI_is_same std::is_same
# else
#   define YAMPI_enable_if boost::enable_if_c
#   define YAMPI_is_same boost::is_same
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  class gather
  {
    ::yampi::communicator comm_;
    ::yampi::rank root_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    gather() = delete;
# else
   private:
    gather();

   public:
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
    BOOST_CONSTEXPR gather(::yampi::communicator const comm, ::yampi::rank const root = ::yampi::rank{0}) BOOST_NOEXCEPT_OR_NOTHROW
      : comm_{comm}, root_{root}
    { }
# else
    BOOST_CONSTEXPR gather(::yampi::communicator const comm, ::yampi::rank const root = ::yampi::rank(0)) BOOST_NOEXCEPT_OR_NOTHROW
      : comm_(comm), root_(root)
    { }
# endif

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    gather(gather const&) = default;
    gather& operator=(gather const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    gather(gather&&) = default;
    gather& operator=(gather&&) = default;
#   endif
    ~gather() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif


    template <typename Value>
    typename YAMPI_enable_if<not ::yampi::is_contiguous_range<Value>::value, void>::type
    call(Value const& send_value) const
    { do_call_value(send_value); }

    template <typename ContiguousRange>
    typename YAMPI_enable_if< ::yampi::is_contiguous_range<ContiguousRange const>::value, void>::type
    call(ContiguousRange const& send_values) const
    { do_call_range(send_values); }


    template <typename Value, typename ContiguousIterator>
    typename YAMPI_enable_if<
      not ::yampi::is_contiguous_iterator<Value>::value
        and not ::yampi::is_contiguous_range<Value>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      void>::type
    call(Value const& send_value, ContiguousIterator const receive_first) const
    { do_call_value(send_value, receive_first); }

    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      void>::type
    call(ContiguousIterator const send_first, int const length) const
    { do_call_iter(send_first, length); }

    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      void>::type
    call(ContiguousIterator const send_first, ContiguousIterator const send_last) const
    { do_call_iter(send_first, send_last); }

    template <typename ContiguousRange, typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange const>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      void>::type
    call(ContiguousRange const& send_values, ContiguousIterator const receive_first) const
    { do_call_range(send_values, receive_first); }


    template <typename ContiguousIterator1, typename ContiguousIterator2>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator1>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator2>::value,
      void>::type
    call(ContiguousIterator1 const send_first, int const length, ContiguousIterator2 const receive_first) const
    { do_call_iter(send_first, length, receive_first); }

    template <typename ContiguousIterator1, typename ContiguousIterator2>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator1>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator2>::value,
      void>::type
    call(ContiguousIterator1 const send_first, ContiguousIterator1 const send_last, ContiguousIterator2 const receive_first) const
    { do_call_iter(send_first, send_last, receive_first); }


   private:
    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    do_call_value(
      typename std::iterator_traits<ContiguousIterator>::value_type const& send_value,
      ContiguousIterator const receive_first) const
    {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
      using value_type = typename std::iterator_traits<ContiguousIterator>::value_type;
# else
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Gather(const_cast<value_type*>(YAMPI_addressof(send_value)), 1, ::yampi::datatype_of<value_type>::call().mpi_datatype(), YAMPI_addressof(*receive_first), 1, ::yampi::datatype_of<value_type>::call().mpi_datatype(), root_.mpi_rank(), comm_.mpi_comm());
# else
      int const error_code
        = MPI_Gather(const_cast<value_type*>(YAMPI_addressof(send_value)), 1, ::yampi::datatype_of<value_type>::call().mpi_datatype(), YAMPI_addressof(*receive_first), 1, ::yampi::datatype_of<value_type>::call().mpi_datatype(), root_.mpi_rank(), comm_.mpi_comm());
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::gather::call"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call");
# endif
    }

    template <typename Value>
    typename YAMPI_enable_if< ::yampi::has_corresponding_datatype<Value>::value, void>::type
    do_call_value(Value const& send_value) const
    {
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (comm_.rank() == root_)
        throw ::yampi::nonroot_call_on_root_error{"yampi::gather::call"};
# else
      if (comm_.rank() == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::gather::call");
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto null = Value{};
#   else
      auto null = Value();
#   endif
# else
      Value null;
# endif

      do_call_value(send_value, YAMPI_addressof(null));
    }


    template <typename ContiguousIterator1, typename ContiguousIterator2>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<typename std::iterator_traits<ContiguousIterator1>::value_type>::value
        and YAMPI_is_same<typename std::iterator_traits<ContiguousIterator1>::value_type, typename std::iterator_traits<ContiguousIterator2>::value_type>::value,
      void>::type
    do_call_iter(ContiguousIterator1 const send_first, int const length, ContiguousIterator2 const receive_first) const
    {
# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
      using value_type = typename std::iterator_traits<ContiguousIterator1>::value_type;
# else
      typedef typename std::iterator_traits<ContiguousIterator1>::value_type value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
      auto const error_code
        = MPI_Gather(YAMPI_addressof(*send_first), length, ::yampi::datatype_of<value_type>::call().mpi_datatype(), YAMPI_addressof(*receive_first), length, ::yampi::datatype_of<value_type>::call().mpi_datatype(), root_.mpi_rank(), comm_.mpi_comm());
# else
      int const error_code
        = MPI_Gather(YAMPI_addressof(*send_first), length, ::yampi::datatype_of<value_type>::call().mpi_datatype(), YAMPI_addressof(*receive_first), length, ::yampi::datatype_of<value_type>::call().mpi_datatype(), root_.mpi_rank(), comm_.mpi_comm());
# endif

# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "mpi::gather"};
# else
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "mpi::gather");
# endif
    }

    template <typename ContiguousIterator>
    void do_call_iter(ContiguousIterator const send_first, int const length) const
    {
# ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      if (comm_.rank() == root_)
        throw ::yampi::nonroot_call_on_root_error{"yampi::gather::call"};
# else
      if (comm_.rank() == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::gather::call");
# endif

# ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
      using value_type = typename std::iterator_traits<ContiguousIterator>::value_type;
# else
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
# endif

# ifndef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
      auto null = value_type{};
#   else
      auto null = value_type();
#   endif
# else
      value_type null;
# endif

      do_call_iter(send_first, length, YAMPI_addressof(null));
    }

    template <typename ContiguousIterator1, typename ContiguousIterator2>
    void do_call_iter(ContiguousIterator1 const send_first, ContiguousIterator1 const send_last, ContiguousIterator2 const receive_first) const
    {
      assert(send_last >= send_first);
      do_call_iter(send_first, send_last-send_first, receive_first);
    }

    template <typename ContiguousIterator>
    void do_call_iter(ContiguousIterator const send_first, ContiguousIterator const send_last) const
    {
      assert(send_last >= send_first);
      do_call_iter(send_first, send_last-send_first);
    }

    template <typename ContiguousRange, typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<typename boost::range_value<ContiguousRange const>::type>::value
        and YAMPI_is_same<typename boost::range_value<ContiguousRange const>::type, typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    do_call_range(ContiguousRange const& send_values, ContiguousIterator const receive_first) const
    {
      using boost::begin;
      using boost::end;
      do_call_iter(begin(send_values), end(send_values), receive_first);
    }

    template <typename ContiguousRange>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<typename boost::range_value<ContiguousRange const>::type>::value,
      void>::type
    do_call_range(ContiguousRange const& send_values) const
    {
      using boost::begin;
      using boost::end;
      do_call_iter(begin(send_values), end(send_values));
    }
  };
}


# undef YAMPI_enable_if
# undef YAMPI_is_same
# undef YAMPI_addressof

#endif

