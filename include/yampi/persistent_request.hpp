#ifndef YAMPI_PERSISTENT_REQUEST_HPP
# define YAMPI_PERSISTENT_REQUEST_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/has_nothrow_copy.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/communicator.hpp>
# include <yampi/communication_mode.hpp>
# include <yampi/message.hpp>
# include <yampi/request_base.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
# else
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  class persistent_request_ref;
  class persistent_request_cref;

  class persistent_request
    : public ::yampi::request_base
  {
    typedef ::yampi::request_base base_type;
    friend class ::yampi::persistent_request_ref;

   public:
    typedef ::yampi::persistent_request_ref reference_type;
    typedef ::yampi::persistent_request_cref const_reference_type;

    persistent_request() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<base_type>::value)
      : base_type()
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    persistent_request(persistent_request const&) = default;
    persistent_request& operator=(persistent_request const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    persistent_request(persistent_request&&) = default;
    persistent_request& operator=(persistent_request&&) = default;
#   endif
    ~persistent_request() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    template <typename Value>
    persistent_request(
      ::yampi::request_send_t const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_standard_send_request(buffer, destination, tag, communicator, environment))
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_send_t const,
      ::yampi::mode::standard_communication const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_standard_send_request(buffer, destination, tag, communicator, environment))
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_send_t const,
      ::yampi::mode::buffered_communication const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_buffered_send_request(buffer, destination, tag, communicator, environment))
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_send_t const,
      ::yampi::mode::synchronous_communication const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_synchronous_send_request(buffer, destination, tag, communicator, environment))
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_send_t const,
      ::yampi::mode::ready_communication const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_ready_send_request(buffer, destination, tag, communicator, environment))
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value>& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_receive_request(buffer, source, tag, communicator, environment))
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value>& buffer, ::yampi::rank const& source,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_receive_request(buffer, source, ::yampi::any_tag(), communicator, environment))
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value>& buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_receive_request(buffer, ::yampi::any_source(), ::yampi::any_tag(), communicator, environment))
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_receive_request(buffer, source, tag, communicator, environment))
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& source,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_receive_request(buffer, source, ::yampi::any_tag(), communicator, environment))
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> const& buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_receive_request(buffer, ::yampi::any_source(), ::yampi::any_tag(), communicator, environment))
    { }

   private:
    template <typename Value>
    static void do_standard_send(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Send_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            YAMPI_addressof(mpi_request));
      if (error_code == MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::do_standard_send", environment);
    }

    template <typename Value>
    static MPI_Request make_standard_send_request(
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_standard_send(result, buffer, destination, tag, communicator, environment);
      return result;
    }

    template <typename Value>
    static void do_buffered_send(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Bsend_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            YAMPI_addressof(mpi_request));
      if (error_code == MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::do_buffered_send", environment);
    }

    template <typename Value>
    static MPI_Request make_buffered_send_request(
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_buffered_send(result, buffer, destination, tag, communicator, environment);
      return result;
    }

    template <typename Value>
    static void do_synchronous_send(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Ssend_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            YAMPI_addressof(mpi_request));
      if (error_code == MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::do_synchronous_send", environment);
    }

    template <typename Value>
    static MPI_Request make_synchronous_send_request(
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_synchronous_send(result, buffer, destination, tag, communicator, environment);
      return result;
    }

    template <typename Value>
    static void do_ready_send(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Rsend_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::do_ready_send", environment);
    }

    template <typename Value>
    static MPI_Request make_ready_send_request(
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_ready_send(result, buffer, destination, tag, communicator, environment);
      return result;
    }

    template <typename Value>
    static void do_receive(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value>& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Recv_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::do_receive", environment);
    }

    template <typename Value>
    static void do_receive(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Recv_init(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::do_receive", environment);
    }

    template <typename Value>
    static MPI_Request make_receive_request(
      ::yampi::buffer<Value>& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_receive(result, buffer, source, tag, communicator, environment);
      return result;
    }

    template <typename Value>
    static MPI_Request make_receive_request(
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_receive(result, buffer, source, tag, communicator, environment);
      return result;
    }

   public:
    void start(::yampi::environment const& environment)
    {
      int const error_code = MPI_Start(YAMPI_addressof(mpi_request_));
      if (error_code == MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::start", environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_send_t const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send(buffer, destination, tag, communicator, environment);
    }

    template <typename Mode, typename Value>
    void reset(
      ::yampi::request_send_t const, Mode const mode,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send(mode, buffer, destination, tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value>& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value>& buffer, ::yampi::rank const& source,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value>& buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& source,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> const& buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, communicator, environment);
    }

    template <typename Value>
    void send(
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_standard_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::standard_communication const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_standard_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::buffered_communication const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_buffered_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::synchronous_communication const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_synchronous_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::ready_communication const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_ready_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value>& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_receive(mpi_request_, buffer, source, tag, communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value>& buffer, ::yampi::rank const& source,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_receive(mpi_request_, buffer, source, ::yampi::any_tag(), communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value>& buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_receive(mpi_request_, buffer, ::yampi::any_source(), ::yampi::any_tag(), communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_receive(mpi_request_, buffer, source, tag, communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& source,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_receive(mpi_request_, buffer, source, ::yampi::any_tag(), communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> const& buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_receive(mpi_request_, buffer, ::yampi::any_source(), ::yampi::any_tag(), communicator, environment); }
  };

  class persistent_request_ref
    : public ::yampi::request_ref_base
  {
    typedef ::yampi::request_ref_base base_type;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    persistent_request_ref() = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    persistent_request_ref();

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~persistent_request_ref() BOOST_NOEXCEPT_OR_NOTHROW = default;
# else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~persistent_request_ref() BOOST_NOEXCEPT_OR_NOTHROW { }
# endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    void reset(::yampi::persistent_request&& request, ::yampi::environment const& environment)
    {
      free(environment);
      *mpi_request_ptr_ = std::move(request.mpi_request_);
      request.mpi_request_ = MPI_REQUEST_NULL;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    void start(::yampi::environment const& environment)
    {
      int const error_code = MPI_Start(mpi_request_ptr_);
      if (error_code == MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::start", environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_send_t const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send(buffer, destination, tag, communicator, environment);
    }

    template <typename Mode, typename Value>
    void reset(
      ::yampi::request_send_t const, Mode const mode,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send(mode, buffer, destination, tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value>& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value>& buffer, ::yampi::rank const& source,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value>& buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& source,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> const& buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, communicator, environment);
    }

    template <typename Value>
    void send(
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { send(::yampi::mode::standard_communication(), buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::standard_communication const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Send_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::send", environment);
    }

    template <typename Value>
    void send(
      ::yampi::mode::buffered_communication const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Bsend_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::send", environment);
    }

    template <typename Value>
    void send(
      ::yampi::mode::synchronous_communication const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Ssend_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::send", environment);
    }

    template <typename Value>
    void send(
      ::yampi::mode::ready_communication const,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Rsend_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::send", environment);
    }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value>& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Recv_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::receive", environment);
    }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value>& buffer, ::yampi::rank const& source,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { receive(buffer, source, ::yampi::any_tag(), communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value>& buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { receive(buffer, ::yampi::any_source(), ::yampi::any_tag(), communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& source, ::yampi::tag const& tag,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Recv_init(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::receive", environment);
    }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const& source,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { receive(buffer, source, ::yampi::any_tag(), communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> const& buffer,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { receive(buffer, ::yampi::any_source(), ::yampi::any_tag(), communicator, environment); }
  };

  class persistent_request_cref
    : public ::yampi::request_cref_base
  {
    typedef ::yampi::request_cref_base base_type;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    persistent_request_cref() = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    persistent_request_cref();

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~persistent_request_cref() BOOST_NOEXCEPT_OR_NOTHROW = default;
# else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~persistent_request_cref() BOOST_NOEXCEPT_OR_NOTHROW { }
# endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
  };
}


# undef YAMPI_addressof
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_move_assignable
# undef YAMPI_is_nothrow_move_constructible
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible

#endif

