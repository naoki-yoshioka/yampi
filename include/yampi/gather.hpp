#ifndef YAMPI_GATHER_HPP
# define YAMPI_GATHER_HPP

# include <boost/config.hpp>

/*
# include <cassert>
*/
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/is_same.hpp>
# endif
# include <iterator>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

/*
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/value_type.hpp>
*/

# include <mpi.h>

//
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif
//

/*
# include <yampi/has_corresponding_datatype.hpp>
# include <yampi/is_contiguous_iterator.hpp>
# include <yampi/is_contiguous_range.hpp>
# include <yampi/datatype_of.hpp>
*/
//
# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
//
# include <yampi/communicator.hpp>
# include <yampi/datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>

/*
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_enable_if boost::enable_if_c
# endif
*/

//
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_same std::is_same
# else
#   define YAMPI_is_same boost::is_same
# endif
//

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

//
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif
//


namespace yampi
{
  class gather
  {
    ::yampi::communicator communicator_;
    ::yampi::rank root_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    gather() = delete;
# else
   private:
    gather();

   public:
# endif

    BOOST_CONSTEXPR gather(
      ::yampi::communicator const communicator, ::yampi::rank const root)
      BOOST_NOEXCEPT_OR_NOTHROW
      : communicator_(communicator), root_(root)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    gather(gather const&) = default;
    gather& operator=(gather const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    gather(gather&&) = default;
    gather& operator=(gather&&) = default;
#   endif
    ~gather() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif


    template <typename SendValue, typename ContiguousIterator>
    void call(
      ::yampi::environment const& environment,
      ::yampi::buffer<SendValue> const& send_buffer,
      ContiguousIterator const out) const
    {
      static_assert(
        (YAMPI_is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");

      int const error_code
        = MPI_Gather(
            const_cast<SendValue*>(send_buffer.data()),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            const_cast<SendValue*>(YAMPI_addressof(*out)),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call", environment);
    }

    template <typename SendValue>
    void call(
      ::yampi::environment const& environment,
      ::yampi::buffer<SendValue> const& send_buffer) const
    {
      if (communicator_.rank(environment) == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::gather::call");

      SendValue null;
      call(environment, send_buffer, YAMPI_addressof(null));
    }
    /*
    template <typename Value, typename ContiguousIterator>
    typename YAMPI_enable_if<
      not ::yampi::is_contiguous_iterator<Value>::value
        and not ::yampi::is_contiguous_range<Value>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      void>::type
    call(Value const& send_value, ContiguousIterator const receive_first) const
    { do_call(send_value, receive_first); }

    template <typename Value>
    typename YAMPI_enable_if<
      not ::yampi::is_contiguous_range<Value const>::value,
      void>::type
    call(Value const& send_value) const
    { do_call(send_value); }


    template <typename ContiguousIterator1, typename ContiguousIterator2>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator1>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator2>::value,
      void>::type
    call(
      ContiguousIterator1 const send_first, ContiguousIterator1 const send_last,
      ContiguousIterator2 const receive_first) const
    { do_call(send_first, send_last, receive_first); }

    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      void>::type
    call(ContiguousIterator const send_first, ContiguousIterator const send_last) const
    { do_call(send_first, send_last); }


    template <typename ContiguousRange, typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange const>::value
        and ::yampi::is_contiguous_iterator<ContiguousIterator>::value,
      void>::type
    call(ContiguousRange const& send_values, ContiguousIterator const receive_first) const
    { do_call(boost::begin(send_values), boost::end(send_values), receive_first); }

    template <typename ContiguousRange>
    typename YAMPI_enable_if<
      ::yampi::is_contiguous_range<ContiguousRange const>::value,
      void>::type
    call(ContiguousRange const& send_values) const
    { do_call(boost::begin(send_values), boost::end(send_values)); }


   private:
    template <typename ContiguousIterator>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<
        typename std::iterator_traits<ContiguousIterator>::value_type>::value,
      void>::type
    do_call(
      typename std::iterator_traits<ContiguousIterator>::value_type const& send_value,
      ContiguousIterator const receive_first) const
    {
      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
      int const error_code
        = MPI_Gather(
            const_cast<value_type*>(YAMPI_addressof(send_value)), 1,
            ::yampi::datatype_of<value_type>::call().mpi_datatype(),
            YAMPI_addressof(*receive_first), 1,
            ::yampi::datatype_of<value_type>::call().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call");
    }

    template <typename Value>
    typename YAMPI_enable_if<
      ::yampi::has_corresponding_datatype<Value>::value,
      void>::type
    do_call(Value const& send_value) const
    {
      if (communicator_.rank() == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::gather::call");

      Value null;
      do_call(send_value, YAMPI_addressof(null));
    }


    template <typename ContiguousIterator1, typename ContiguousIterator2>
    void do_call(
      ContiguousIterator1 const send_first, ContiguousIterator1 const send_last,
      ContiguousIterator2 const receive_first) const
    {
      assert(send_last >= send_first);

      typedef typename std::iterator_traits<ContiguousIterator1>::value_type value_type;
      int const error_code
        = MPI_Gather(
            YAMPI_addressof(*send_first), send_last-send_first,
            ::yampi::datatype_of<value_type>::call().mpi_datatype(),
            YAMPI_addressof(*receive_first), send_last-send_first,
            ::yampi::datatype_of<value_type>::call().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "mpi::gather");
    }

    template <typename ContiguousIterator>
    void do_call(
      ContiguousIterator const send_first, ContiguousIterator const send_last) const
    {
      if (communicator_.rank() == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::gather::call");

      typedef typename std::iterator_traits<ContiguousIterator>::value_type value_type;
      value_type null;
      do_call(send_first, send_last, YAMPI_addressof(null));
    }
    */
  };
}


//
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
//
/*
# undef YAMPI_enable_if
*/
# undef YAMPI_addressof
# undef YAMPI_is_same

#endif

