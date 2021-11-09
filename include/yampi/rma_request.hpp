#ifndef YAMPI_RMA_REQUEST_HPP
# define YAMPI_RMA_REQUEST_HPP

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

# include <yampi/window_base.hpp>
# include <yampi/buffer.hpp>
# include <yampi/target_buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/binary_operation.hpp>
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


# if MPI_VERSION >= 3
namespace yampi
{
  struct request_put_t { };
  struct request_get_t { };
  struct request_accumulate_t { };
  struct request_fetch_accumulate_t { };

  inline BOOST_CONSTEXPR ::yampi::request_put_t request_put() BOOST_NOEXCEPT_OR_NOTHROW
  { return ::yampi::request_put_t(); }

  inline BOOST_CONSTEXPR ::yampi::request_get_t request_get() BOOST_NOEXCEPT_OR_NOTHROW
  { return ::yampi::request_get_t(); }

  inline BOOST_CONSTEXPR ::yampi::request_accumulate_t request_accumulate() BOOST_NOEXCEPT_OR_NOTHROW
  { return ::yampi::request_accumulate_t(); }

  inline BOOST_CONSTEXPR ::yampi::request_fetch_accumulate_t request_fetch_accumulate() BOOST_NOEXCEPT_OR_NOTHROW
  { return ::yampi::request_fetch_accumulate_t(); }

  class rma_request_ref;
  class rma_request_cref;

  class rma_request
    : public ::yampi::request_base
  {
    typedef ::yampi::request_base base_type;
    friend class ::yampi::rma_request_ref;

   public:
    typedef ::yampi::rma_request_ref reference_type;
    typedef ::yampi::rma_request_cref const_reference_type;

    rma_request() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<base_type>::value)
      : base_type()
    { }

#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    rma_request(rma_request const&) = default;
    rma_request& operator=(rma_request const&) = default;
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    rma_request(rma_request&&) = default;
    rma_request& operator=(rma_request&&) = default;
#     endif
    ~rma_request() BOOST_NOEXCEPT_OR_NOTHROW = default;
#   endif

    template <typename OriginValue, typename TargetValue, typename Window>
    rma_request(
      ::yampi::request_put_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
      : base_type(make_put_request(origin_buffer, target, target_buffer, window, environment))
    { }

    template <typename OriginValue, typename TargetValue, typename Window>
    rma_request(
      ::yampi::request_get_t const,
      ::yampi::buffer<OriginValue>& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
      : base_type(make_get_request(origin_buffer, target, target_buffer, window, environment))
    { }

    template <typename OriginValue, typename TargetValue, typename Window>
    rma_request(
      ::yampi::request_get_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
      : base_type(make_get_request(origin_buffer, target, target_buffer, window, environment))
    { }

    template <typename OriginValue, typename TargetValue, typename Window>
    rma_request(
      ::yampi::request_accumulate_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
      : base_type(make_accumulate_request(origin_buffer, target, target_buffer, operation, window, environment))
    { }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    rma_request(
      ::yampi::request_fetch_accumulate_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue>& result_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
      : base_type(make_fetch_accumulate_request(origin_buffer, result_buffer, target, target_buffer, operation, window, environment))
    { }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    rma_request(
      ::yampi::request_fetch_accumulate_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue> const& result_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
      : base_type(make_fetch_accumulate_request(origin_buffer, result_buffer, target, target_buffer, operation, window, environment))
    { }

   private:
    template <typename OriginValue, typename TargetValue, typename Window>
    static void do_put(
      MPI_Request& mpi_request,
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Rput(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request::do_put", environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    static MPI_Request make_put_request(
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_put(result, origin_buffer, target, target_buffer, window, environment);
      return result;
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    static void do_get(
      MPI_Request& mpi_request,
      ::yampi::buffer<OriginValue>& origin_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Rget(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request::do_get", environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    static void do_get(
      MPI_Request& mpi_request,
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Rget(
            const_cast<OriginValue*>(origin_buffer.data()), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request::do_get", environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    static MPI_Request make_get_request(
      ::yampi::buffer<OriginValue>& origin_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_get(result, origin_buffer, target, target_buffer, window, environment);
      return result;
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    static MPI_Request make_get_request(
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_get(result, origin_buffer, target, target_buffer, window, environment);
      return result;
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    static void do_accumulate(
      MPI_Request& mpi_request,
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Raccumulate(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request::do_accumulate", environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    static MPI_Request make_accumulate_request(
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_accumulate(result, origin_buffer, target, target_buffer, operation, window, environment);
      return result;
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    static void do_fetch_accumulate(
      MPI_Request& mpi_request,
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue>& result_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Rget_accumulate(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            result_buffer.data(), result_buffer.count(), result_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request::do_fetch_accumulate", environment);
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    static void do_fetch_accumulate(
      MPI_Request& mpi_request,
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue> const& result_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Rget_accumulate(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            const_cast<ResultValue*>(result_buffer.data()), result_buffer.count(), result_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request::do_fetch_accumulate", environment);
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    static MPI_Request make_fetch_accumulate_request(
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue>& result_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_fetch_accumulate(result, origin_buffer, result_buffer, target, target_buffer, operation, window, environment);
      return result;
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    static MPI_Request make_fetch_accumulate_request(
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue> const& result_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_fetch_accumulate(result, origin_buffer, result_buffer, target, target_buffer, operation, window, environment);
      return result;
    }

   public:
    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_put_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      put(origin_buffer, target, target_buffer, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_get_t const,
      ::yampi::buffer<OriginValue>& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      get(origin_buffer, target, target_buffer, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_get_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      get(origin_buffer, target, target_buffer, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_accumulate_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      accumulate(origin_buffer, target, target_buffer, operation, window, environment);
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_fetch_accumulate_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue>& result_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      fetch_accumulate(origin_buffer, result_buffer, target, target_buffer, operation, window, environment);
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_fetch_accumulate_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue> const& result_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      fetch_accumulate(origin_buffer, result_buffer, target, target_buffer, operation, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void put(
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    { do_put(mpi_request_, origin_buffer, target, target_buffer, window, environment); }

    template <typename OriginValue, typename TargetValue, typename Window>
    void get(
      ::yampi::buffer<OriginValue>& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    { do_get(mpi_request_, origin_buffer, target, target_buffer, window, environment); }

    template <typename OriginValue, typename TargetValue, typename Window>
    void get(
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    { do_get(mpi_request_, origin_buffer, target, target_buffer, window, environment); }

    template <typename OriginValue, typename TargetValue, typename Window>
    void accumulate(
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    { do_accumulate(mpi_request_, origin_buffer, target, target_buffer, operation, window, environment); }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    void fetch_accumulate(
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue>& result_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    { do_fetch_accumulate(mpi_request_, origin_buffer, result_buffer, target, target_buffer, operation, window, environment); }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    void fetch_accumulate(
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue> const& result_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    { do_fetch_accumulate(mpi_request_, origin_buffer, result_buffer, target, target_buffer, operation, window, environment); }
  };

  class rma_request_ref
    : public ::yampi::request_ref_base
  {
    typedef ::yampi::request_ref_base base_type;

   public:
#   ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    rma_request_ref() = delete;
#   else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    rma_request_ref();

   public:
#   endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~rma_request_ref() BOOST_NOEXCEPT_OR_NOTHROW = default;
#   else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~rma_request_ref() BOOST_NOEXCEPT_OR_NOTHROW { }
#   endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS

#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    void reset(::yampi::rma_request&& request, ::yampi::environment const& environment)
    {
      free(environment);
      *mpi_request_ptr_ = std::move(request.mpi_request_);
      request.mpi_request_ = MPI_REQUEST_NULL;
    }
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_put_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      put(origin_buffer, target, target_buffer, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_get_t const,
      ::yampi::buffer<OriginValue>& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      get(origin_buffer, target, target_buffer, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_get_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      get(origin_buffer, target, target_buffer, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_accumulate_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      accumulate(origin_buffer, target, target_buffer, operation, window, environment);
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_fetch_accumulate_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue>& result_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      fetch_accumulate(origin_buffer, result_buffer, target, target_buffer, operation, window, environment);
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_fetch_accumulate_t const,
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue> const& result_buffer,
      ::yampi::rank const& target, ::yampi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      fetch_accumulate(origin_buffer, result_buffer, target, target_buffer, operation, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void put(
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Rput(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request_ref::put", environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void get(
      ::yampi::buffer<OriginValue>& origin_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Rget(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request_ref::get", environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void get(
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Rget(
            const_cast<OriginValue*>(origin_buffer.data()), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request_ref::get", environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void accumulate(
      ::yampi::buffer<OriginValue> const& origin_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Raccumulate(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request_ref::accumulate", environment);
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    void fetch_accumulate(
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue>& result_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Rget_accumulate(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            result_buffer.data(), result_buffer.count(), result_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request_ref::fetch_accumulate", environment);
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    void fetch_accumulate(
      ::yampi::buffer<OriginValue> const& origin_buffer, ::yampi::buffer<ResultValue> const& result_buffer,
      ::yampi::rank const& target, ::yambi::target_buffer<TargetValue> const& target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Rget_accumulate(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            const_cast<ResultValue*>(result_buffer.data()), result_buffer.count(), result_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request_ref::fetch_accumulate", environment);
    }
  };

  class rma_request_cref
    : public ::yampi::request_cref_base
  {
    typedef ::yampi::request_cref_base base_type;

   public:
#   ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    rma_request_cref() = delete;
#   else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    rma_request_cref();

   public:
#   endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~rma_request_cref() BOOST_NOEXCEPT_OR_NOTHROW = default;
#   else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~rma_request_cref() BOOST_NOEXCEPT_OR_NOTHROW { }
#   endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
}
# endif // MPI_VERSION >= 3


# undef YAMPI_addressof
# undef YAMPI_is_nothrow_copy_constructible

#endif

