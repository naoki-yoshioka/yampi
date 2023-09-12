#ifndef YAMPI_PERSISTENT_REQUEST_HPP
# define YAMPI_PERSISTENT_REQUEST_HPP

# include <utility>
# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/communication_mode.hpp>
# include <yampi/message.hpp>
# include <yampi/request_base.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


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

    persistent_request() noexcept(std::is_nothrow_copy_constructible<base_type>::value)
      : base_type{}
    { }

    persistent_request(persistent_request const&) = delete;
    persistent_request& operator=(persistent_request const&) = delete;
    persistent_request(persistent_request&&) = default;
    persistent_request& operator=(persistent_request&&) = default;
    ~persistent_request() noexcept = default;

    using base_type::base_type;

    // send
    template <typename Value>
    persistent_request(
      ::yampi::request_send_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_standard_send_request(buffer, destination, tag, communicator, environment)}
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_send_t const,
      ::yampi::mode::standard_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_standard_send_request(buffer, destination, tag, communicator, environment)}
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_send_t const,
      ::yampi::mode::buffered_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_buffered_send_request(buffer, destination, tag, communicator, environment)}
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_send_t const,
      ::yampi::mode::synchronous_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_synchronous_send_request(buffer, destination, tag, communicator, environment)}
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_send_t const,
      ::yampi::mode::ready_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_ready_send_request(buffer, destination, tag, communicator, environment)}
    { }

    // receive
    template <typename Value>
    persistent_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_receive_request(buffer, source, tag, communicator, environment)}
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_receive_request(buffer, source, ::yampi::any_tag, communicator, environment)}
    { }

    template <typename Value>
    persistent_request(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      : base_type{make_receive_request(buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment)}
    { }

   private:
    // send
    template <typename Value>
    static void do_standard_send(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Send_init_c(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# elif MPI_VERSION >= 3
      int const error_code
        = MPI_Send_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# else // MPI_VERSION >= 3
      int const error_code
        = MPI_Send_init(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# endif // MPI_VERSION >= 3
      if (error_code == MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::do_standard_send", environment);
    }

    template <typename Value>
    static MPI_Request make_standard_send_request(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_standard_send(result, buffer, destination, tag, communicator, environment);
      return result;
    }

    template <typename Value>
    static void do_buffered_send(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Bsend_init_c(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# elif MPI_VERSION >= 3
      int const error_code
        = MPI_Bsend_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# else // MPI_VERSION >= 3
      int const error_code
        = MPI_Bsend_init(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# endif // MPI_VERSION >= 3
      if (error_code == MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::do_buffered_send", environment);
    }

    template <typename Value>
    static MPI_Request make_buffered_send_request(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_buffered_send(result, buffer, destination, tag, communicator, environment);
      return result;
    }

    template <typename Value>
    static void do_synchronous_send(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Ssend_init_c(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# elif MPI_VERSION >= 3
      int const error_code
        = MPI_Ssend_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# else // MPI_VERSION >= 3
      int const error_code
        = MPI_Ssend_init(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# endif // MPI_VERSION >= 3
      if (error_code == MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::do_synchronous_send", environment);
    }

    template <typename Value>
    static MPI_Request make_synchronous_send_request(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_synchronous_send(result, buffer, destination, tag, communicator, environment);
      return result;
    }

    template <typename Value>
    static void do_ready_send(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Rsend_init_c(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# elif MPI_VERSION >= 3
      int const error_code
        = MPI_Rsend_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# else // MPI_VERSION >= 3
      int const error_code
        = MPI_Rsend_init(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# endif // MPI_VERSION >= 3
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::do_ready_send", environment);
    }

    template <typename Value>
    static MPI_Request make_ready_send_request(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_ready_send(result, buffer, destination, tag, communicator, environment);
      return result;
    }

    // receive
    template <typename Value>
    static void do_receive(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Recv_init_c(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Recv_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::do_receive", environment);
    }

    template <typename Value>
    static MPI_Request make_receive_request(
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_receive(result, buffer, source, tag, communicator, environment);
      return result;
    }

   public:
    void start(::yampi::environment const& environment)
    {
      int const error_code = MPI_Start(std::addressof(mpi_request_));
      if (error_code == MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request::start", environment);
    }

    using base_type::reset;

    // send
    template <typename Value>
    void reset(
      ::yampi::request_send_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send(buffer, destination, tag, communicator, environment);
    }

    template <typename Mode, typename Value>
    void reset(
      ::yampi::request_send_t const, Mode const mode,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send(mode, buffer, destination, tag, communicator, environment);
    }

    // receive
    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, communicator, environment);
    }

    // send
    template <typename Value>
    void send(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { do_standard_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::standard_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { do_standard_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::buffered_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { do_buffered_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::synchronous_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { do_synchronous_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::ready_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { do_ready_send(mpi_request_, buffer, destination, tag, communicator, environment); }

    // receive
    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { do_receive(mpi_request_, buffer, source, tag, communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { do_receive(mpi_request_, buffer, source, ::yampi::any_tag, communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { do_receive(mpi_request_, buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment); }
  };

  inline void swap(::yampi::persistent_request& lhs, ::yampi::persistent_request& rhs) noexcept
  { lhs.swap(rhs); }

  class persistent_request_ref
    : public ::yampi::request_ref_base
  {
    typedef ::yampi::request_ref_base base_type;

   public:
    persistent_request_ref() = delete;
    ~persistent_request_ref() noexcept = default;

    using base_type::base_type;

    void start(::yampi::environment const& environment)
    {
      int const error_code = MPI_Start(mpi_request_ptr_);
      if (error_code == MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::start", environment);
    }

    using base_type::reset;

    void reset(::yampi::persistent_request&& request, ::yampi::environment const& environment)
    {
      free(environment);
      *mpi_request_ptr_ = std::move(request.mpi_request_);
      request.mpi_request_ = MPI_REQUEST_NULL;
    }

    // send
    template <typename Value>
    void reset(
      ::yampi::request_send_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send(buffer, destination, tag, communicator, environment);
    }

    template <typename Mode, typename Value>
    void reset(
      ::yampi::request_send_t const, Mode const mode,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      send(mode, buffer, destination, tag, communicator, environment);
    }

    // receive
    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, tag, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, source, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::request_receive_t const,
      ::yampi::buffer<Value> buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      receive(buffer, communicator, environment);
    }

    // send
    template <typename Value>
    void send(
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { send(::yampi::mode::standard_communication(), buffer, destination, tag, communicator, environment); }

    template <typename Value>
    void send(
      ::yampi::mode::standard_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Send_init_c(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# elif MPI_VERSION >= 3
      int const error_code
        = MPI_Send_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# else // MPI_VERSION >= 3
      int const error_code
        = MPI_Send_init(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# endif // MPI_VERSION >= 3
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::send", environment);
    }

    template <typename Value>
    void send(
      ::yampi::mode::buffered_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Bsend_init_c(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# elif MPI_VERSION >= 3
      int const error_code
        = MPI_Bsend_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# else // MPI_VERSION >= 3
      int const error_code
        = MPI_Bsend_init(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# endif // MPI_VERSION >= 3
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::send", environment);
    }

    template <typename Value>
    void send(
      ::yampi::mode::synchronous_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Ssend_init_c(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# elif MPI_VERSION >= 3
      int const error_code
        = MPI_Ssend_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# else // MPI_VERSION >= 3
      int const error_code
        = MPI_Ssend_init(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# endif // MPI_VERSION >= 3
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::send", environment);
    }

    template <typename Value>
    void send(
      ::yampi::mode::ready_communication_t const,
      ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Rsend_init_c(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# elif MPI_VERSION >= 3
      int const error_code
        = MPI_Rsend_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# else // MPI_VERSION >= 3
      int const error_code
        = MPI_Rsend_init(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# endif // MPI_VERSION >= 3
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::send", environment);
    }

    // receive
    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Recv_init_c(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Recv_init(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
            mpi_request_ptr_);
# endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::persistent_request_ref::receive", environment);
    }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer, ::yampi::rank const source,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { receive(buffer, source, ::yampi::any_tag, communicator, environment); }

    template <typename Value>
    void receive(
      ::yampi::buffer<Value> buffer,
      ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
    { receive(buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment); }
  };

  inline void swap(::yampi::persistent_request_ref& lhs, ::yampi::persistent_request_ref& rhs) noexcept
  { lhs.swap(rhs); }

  class persistent_request_cref
    : public ::yampi::request_cref_base
  {
    typedef ::yampi::request_cref_base base_type;

   public:
    persistent_request_cref() = delete;
    ~persistent_request_cref() noexcept = default;

    using base_type::base_type;
  };
}


#endif

