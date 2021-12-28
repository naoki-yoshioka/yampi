#ifndef YAMPI_COMPLETE_EXCHANGE_HPP
# define YAMPI_COMPLETE_EXCHANGE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/is_same.hpp>
#   if MPI_VERSION >= 3
#     include <boost/type_traits/has_nothrow_copy.hpp>
#   endif // MPI_VERSION >= 3
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
#   include <yampi/topology.hpp>
# endif
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_same std::is_same
#   if MPI_VERSION >= 3
#     define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   endif // MPI_VERSION >= 3
# else
#   define YAMPI_is_same boost::is_same
#   if MPI_VERSION >= 3
#     define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   endif // MPI_VERSION >= 3
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
  // TODO: implement MPI_Alltoallv, MPI_Alltoallw
  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 3
    int const error_code
      = MPI_Alltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Alltoall(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::complete_exchange", environment);
  }

  template <typename Value>
  inline void complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value> receive_buffer,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Alltoall(
          MPI_IN_PLACE, receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::complete_exchange", environment);
  }
# if MPI_VERSION >= 3

  // neighbor complete_exchange
  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::topology const& topology,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Neighbor_alltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::complete_exchange", environment);
  }

  template <typename Value>
  inline void complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value> receive_buffer,
    ::yampi::topology const& topology,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Neighbor_alltoall(
          MPI_IN_PLACE, receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::complete_exchange", environment);
  }

  class complete_exchange_request
    : public ::yampi::request_base
  {
    typedef request_base base_type;

   public:
    complete_exchange_request() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<base_type>::value)
      : base_type()
    { }

#   ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    complete_exchange_request(complete_exchange_request const&) = delete;
    complete_exchange_request& operator=(complete_exchange_request const&) = delete;
#   else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    complete_exchange_request(complete_exchange_request const&);
    complete_exchange_request& operator=(complete_exchange_request const&);

   public:
#   endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    complete_exchange_request(complete_exchange_request&&) = default;
    complete_exchange_request& operator=(complete_exchange_request&&) = default;
#     endif
    ~complete_exchange_request() BOOST_NOEXCEPT_OR_NOTHROW = default;
#   endif

    // complete_exchange
    template <typename SendValue, typename ReceiveValue>
    complete_exchange_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_complete_exchange_request(send_buffer, receive_buffer, communicator, environment))
    { }

    template <typename Value>
    complete_exchange_request(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_complete_exchange_in_place_request(receive_buffer, communicator, environment))
    { }

    // neighbor complete_exchange
    template <typename SendValue, typename ReceiveValue>
    complete_exchange_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology const& topology, ::yampi::environment const& environment)
      : base_type(make_complete_exchange_request(send_buffer, receive_buffer, topology, environment))
    { }

    template <typename Value>
    complete_exchange_request(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::topology const& topology, ::yampi::environment const& environment)
      : base_type(make_complete_exchange_in_place_request(receive_buffer, topology, environment))
    { }

   private:
    // complete_exchange
    template <typename SendValue, typename ReceiveValue>
    static void do_complete_exchange(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Ialltoall(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::complete_exchange_request::do_complete_exchange", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    static MPI_Request make_complete_exchange_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_complete_exchange(result, send_buffer, receive_buffer, communicator, environment);
      return result;
    }

    template <typename Value>
    static void do_complete_exchange_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Ialltoall(
            MPI_IN_PLACE, receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::complete_exchange_request::do_complete_exchange_in_place", environment);
    }

    template <typename Value>
    static MPI_Request make_complete_exchange_in_place_request(
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_complete_exchange_in_place(result, receive_buffer, communicator, environment);
      return result;
    }

    // neighbor complete_exchange
    template <typename SendValue, typename ReceiveValue>
    static void do_complete_exchange(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology const& topology, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Ineighbor_alltoall(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::complete_exchange_request::do_complete_exchange", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    static MPI_Request make_complete_exchange_request(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology const& topology, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_complete_exchange(result, send_buffer, receive_buffer, topology, environment);
      return result;
    }

    template <typename Value>
    static void do_complete_exchange_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::topology const& topology, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Ineighbor_alltoall(
            MPI_IN_PLACE, receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            topology.communicator().mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::complete_exchange_request::do_complete_exchange_in_place", environment);
    }

    template <typename Value>
    static MPI_Request make_complete_exchange_in_place_request(
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::topology const& topology, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_complete_exchange_in_place_request(result, receive_buffer, topology, environment);
      return result;
    }

   public:
    // complete_exchange
    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, receive_buffer, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(in_place, receive_buffer, communicator, environment);
    }

    // neighbor complete_exchange
    template <typename SendValue, typename ReceiveValue>
    void reset(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(send_buffer, receive_buffer, topology, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::topology const& topology, ::yampi::environment const& environment)
    {
      free(environment);
      complete_exchange(in_place, receive_buffer, topology, environment);
    }

    // complete_exchange
    template <typename SendValue, typename ReceiveValue>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_complete_exchange(mpi_request_, send_buffer, receive_buffer, communicator, environment); }

    template <typename Value>
    void complete_exchange(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_complete_exchange_in_place(mpi_request_, receive_buffer, communicator, environment); }

    // neighbor complete_exchange
    template <typename SendValue, typename ReceiveValue>
    void complete_exchange(
      ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::topology const& topology, ::yampi::environment const& environment)
    { do_complete_exchange(mpi_request_, send_buffer, receive_buffer, topology, environment); }

    template <typename Value>
    void complete_exchange(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::topology const& topology, ::yampi::environment const& environment)
    { do_complete_exchange_in_place(mpi_request_, receive_buffer, topology, environment); }
  };
# endif // MPI_VERSION >= 3
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# if MPI_VERSION >= 3
#   undef YAMPI_is_nothrow_copy_constructible
# endif
# undef YAMPI_is_same

#endif
