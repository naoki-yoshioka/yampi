#ifndef YAMPI_GATHER_HPP
# define YAMPI_GATHER_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/is_same.hpp>
#   include <boost/type_traits/has_nothrow_copy.hpp>
# endif
# include <iterator>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/in_place.hpp>
# if MPI_VERSION >= 3
#   include <yampi/request_base.hpp>
# endif
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>
# include <yampi/root_call_on_nonroot_error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_same std::is_same
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
# else
#   define YAMPI_is_same boost::is_same
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif


namespace yampi
{
  // TODO: implement MPI_Gatherv
  class gather
  {
    ::yampi::rank root_;
    ::yampi::communicator const& communicator_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    gather() = delete;
    gather(gather const&) = delete;
    gather& operator=(gather const&) = delete;
# else
   private:
    gather();
    gather(gather const&);
    gather& operator=(gather const&);

   public:
# endif

    gather(
      ::yampi::rank const& root, ::yampi::communicator const& communicator)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible< ::yampi::rank >::value)
      : root_(root), communicator_(communicator)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    gather(gather&&) = default;
    gather& operator=(gather&&) = default;
#   endif
    ~gather() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif


    template <typename SendValue, typename ContiguousIterator>
    void call(
      ::yampi::buffer<SendValue> const& send_buffer,
      ContiguousIterator const first,
      ::yampi::environment const& environment) const
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
            const_cast<SendValue*>(YAMPI_addressof(*first)),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void call(
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::buffer<ReceiveValue>& receive_buffer,
      ::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Gather(
            const_cast<SendValue*>(send_buffer.data()),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void call(
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::buffer<ReceiveValue> const& receive_buffer,
      ::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Gather(
            const_cast<SendValue*>(send_buffer.data()),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            const_cast<ReceiveValue*>(receive_buffer.data()),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call", environment);
    }

    template <typename SendValue>
    void call(
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::environment const& environment) const
    {
      if (communicator_.rank(environment) == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::gather::call");

      SendValue null;
      call(send_buffer, YAMPI_addressof(null), environment);
    }

    template <typename Value>
    void call(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value>& receive_buffer,
      ::yampi::environment const& environment) const
    {
      if (communicator_.rank(environment) != root_)
        throw ::yampi::root_call_on_nonroot_error("yampi::gather::call");

      int const error_code
        = MPI_Gather(
            MPI_IN_PLACE, receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            receive_buffer.data(),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call", environment);
    }

    template <typename Value>
    void call(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> const& receive_buffer,
      ::yampi::environment const& environment) const
    {
      if (communicator_.rank(environment) != root_)
        throw ::yampi::root_call_on_nonroot_error("yampi::gather::call");

      int const error_code
        = MPI_Gather(
            MPI_IN_PLACE, receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            const_cast<Value*>(receive_buffer.data()),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call", environment);
    }
  };
# if MPI_VERSION >= 3

  class gather_request
    : public ::yampi::request_base
  {
    typedef request_base base_type;

   public:
    gather_request() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<base_type>::value)
      : base_type()
    { }

#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    gather_request(gather_request const&) = default;
    gather_request& operator=(gather_request const&) = default;
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    gather_request(gather_request&&) = default;
    gather_request& operator=(gather_request&&) = default;
#     endif
    ~gather_request() BOOST_NOEXCEPT_OR_NOTHROW = default;
#   endif

    template <typename SendValue, typename ContiguousIterator>
    gather_request(
      ::yampi::buffer<SendValue> const& send_buffer, ContiguousIterator const first, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_gather_request(send_buffer, first, root, communicator, environment))
    { }

    template <typename SendValue, typename ReceiveValue>
    gather_request(
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::buffer<ReceiveValue>& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_gather_request(send_buffer, receive_buffer, root, communicator, environment))
    { }

    template <typename SendValue, typename ReceiveValue>
    gather_request(
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::buffer<ReceiveValue> const& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_gather_request(send_buffer, receive_buffer, root, communicator, environment))
    { }

    template <typename SendValue>
    gather_request(
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_gather_request(send_buffer, root, communicator, environment))
    { }

    template <typename Value>
    gather_request(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value>& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_gather_in_place_request(receive_buffer, root, communicator, environment))
    { }

    template <typename Value>
    gather_request(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> const& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_gather_in_place_request(receive_buffer, root, communicator, environment))
    { }

   private:
    template <typename SendValue, typename ContiguousIterator>
    static void do_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const& send_buffer, ContiguousIterator const first, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (YAMPI_is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");

      int const error_code
        = MPI_Igather(
            const_cast<SendValue*>(send_buffer.data()),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            const_cast<SendValue*>(YAMPI_addressof(*first)),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather_request::do_gather", environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    static MPI_Request make_gather_request(
      ::yampi::buffer<SendValue> const& send_buffer, ContiguousIterator const first, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_gather(result, send_buffer, first, root, communicator, environment);
      return result;
    }

    template <typename SendValue, typename ReceiveValue>
    static void do_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::buffer<ReceiveValue>& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Igather(
            const_cast<SendValue*>(send_buffer.data()),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather_request::do_gather", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    static void do_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::buffer<ReceiveValue> const& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Igather(
            const_cast<SendValue*>(send_buffer.data()),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            const_cast<ReceiveValue*>(receive_buffer.data()),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather_request::do_gather", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    static MPI_Request make_gather_request(
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::buffer<ReceiveValue>& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_gather(result, send_buffer, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename SendValue, typename ReceiveValue>
    static MPI_Request make_gather_request(
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::buffer<ReceiveValue> const& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_gather(result, send_buffer, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename SendValue>
    static void do_gather(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) == root)
        throw ::yampi::nonroot_call_on_root_error("yampi::gather_request::do_gather");

      SendValue null;
      do_gather(mpi_request, send_buffer, YAMPI_addressof(null), root, communicator, environment);
    }

    template <typename SendValue>
    static MPI_Request make_gather_request(
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_gather(result, send_buffer, root, communicator, environment);
      return result;
    }

    template <typename Value>
    static void do_gather_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value>& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) != root)
        throw ::yampi::root_call_on_nonroot_error("yampi::gather_request::do_gather_in_place");

      int const error_code
        = MPI_Igather(
            MPI_IN_PLACE, receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            receive_buffer.data(),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather_request::do_gather_in_place", environment);
    }

    template <typename Value>
    static void do_gather_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) != root)
        throw ::yampi::root_call_on_nonroot_error("yampi::gather_request::do_gather_in_place");

      int const error_code
        = MPI_Igather(
            MPI_IN_PLACE, receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            const_cast<Value*>(receive_buffer.data()),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather_request::do_gather_in_place", environment);
    }

    template <typename Value>
    static MPI_Request make_gather_in_place_request(
      ::yampi::buffer<Value>& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_gather_in_place(result, receive_buffer, root, communicator, environment);
      return result;
    }

    template <typename Value>
    static MPI_Request make_gather_in_place_request(
      ::yampi::buffer<Value> const& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_gather_in_place(result, receive_buffer, root, communicator, environment);
      return result;
    }

   public:
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::buffer<SendValue> const& send_buffer, ContiguousIterator const first, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(send_buffer, first, root, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::buffer<ReceiveValue>& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(send_buffer, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::buffer<ReceiveValue> const& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(send_buffer, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(send_buffer, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value>& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(in_place, receive_buffer, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> const& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      gather(in_place, receive_buffer, root, communicator, environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    void gather(
      ::yampi::buffer<SendValue> const& send_buffer, ContiguousIterator const first, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_gather(mpi_request_, send_buffer, first, root, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void gather(
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::buffer<ReceiveValue>& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_gather(mpi_request_, send_buffer, receive_buffer, root, communicator, environment); }

    template <typename SendValue, typename ReceiveValue>
    void gather(
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::buffer<ReceiveValue> const& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_gather(mpi_request_, send_buffer, receive_buffer, root, communicator, environment); }

    template <typename SendValue>
    void gather(
      ::yampi::buffer<SendValue> const& send_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_gather(mpi_request_, send_buffer, root, communicator, environment); }

    template <typename Value>
    void gather(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value>& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_gather_in_place(mpi_request_, receive_buffer, root, communicator, environment); }

    template <typename Value>
    void gather(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> const& receive_buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_gather_in_place(mpi_request_, receive_buffer, root, communicator, environment); }
  };
# endif // MPI_VERSION >= 3
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_is_nothrow_copy_constructible
# undef YAMPI_is_same

#endif

