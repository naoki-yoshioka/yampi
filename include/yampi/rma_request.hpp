#ifndef YAMPI_RMA_REQUEST_HPP
# define YAMPI_RMA_REQUEST_HPP

# include <cassert>
# include <utility>
# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/buffer.hpp>
# include <yampi/target_buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/request_base.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


# if MPI_VERSION >= 3
namespace yampi
{
  struct request_put_t { };
  struct request_get_t { };
  struct request_accumulate_t { };
  struct request_fetch_accumulate_t { };

# if __cplusplus >= 201703L
  inline constexpr ::yampi::request_put_t request_put{};
  inline constexpr ::yampi::request_get_t request_get{};
  inline constexpr ::yampi::request_accumulate_t request_accumulate{};
  inline constexpr ::yampi::request_fetch_accumulate_t request_fetch_accumulate{};
# else
  constexpr ::yampi::request_put_t request_put{};
  constexpr ::yampi::request_get_t request_get{};
  constexpr ::yampi::request_accumulate_t request_accumulate{};
  constexpr ::yampi::request_fetch_accumulate_t request_fetch_accumulate{};
# endif

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

    rma_request() noexcept(std::is_nothrow_copy_constructible<base_type>::value)
      : base_type{}
    { }

    rma_request(rma_request const&) = delete;
    rma_request& operator=(rma_request const&) = delete;
    rma_request(rma_request&&) = default;
    rma_request& operator=(rma_request&&) = default;
    ~rma_request() noexcept = default;

    using base_type::base_type;

    template <typename OriginValue, typename TargetValue, typename Window>
    rma_request(
      ::yampi::request_put_t const,
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
      : base_type{make_put_request(origin_buffer, target, target_buffer, window, environment)}
    { }

    template <typename OriginValue, typename TargetValue, typename Window>
    rma_request(
      ::yampi::request_get_t const,
      ::yampi::buffer<OriginValue> origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
      : base_type{make_get_request(origin_buffer, target, target_buffer, window, environment)}
    { }

    template <typename OriginValue, typename TargetValue, typename Window>
    rma_request(
      ::yampi::request_accumulate_t const,
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
      : base_type{make_accumulate_request(origin_buffer, target, target_buffer, operation, window, environment)}
    { }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    rma_request(
      ::yampi::request_fetch_accumulate_t const,
      ::yampi::buffer<OriginValue> const origin_buffer, ::yampi::buffer<ResultValue> result_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
      : base_type{make_fetch_accumulate_request(origin_buffer, result_buffer, target, target_buffer, operation, window, environment)}
    { }

   private:
    template <typename OriginValue, typename TargetValue, typename Window>
    static void do_put(
      MPI_Request& mpi_request,
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Rput_c(
            origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Rput(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request::do_put", environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    static MPI_Request make_put_request(
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_put(result, origin_buffer, target, target_buffer, window, environment);
      return result;
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    static void do_get(
      MPI_Request& mpi_request,
      ::yampi::buffer<OriginValue> origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Rget_c(
            origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Rget(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request::do_get", environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    static MPI_Request make_get_request(
      ::yampi::buffer<OriginValue> origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_get(result, origin_buffer, target, target_buffer, window, environment);
      return result;
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    static void do_accumulate(
      MPI_Request& mpi_request,
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Raccumulate_c(
            origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Raccumulate(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request::do_accumulate", environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    static MPI_Request make_accumulate_request(
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
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
      ::yampi::buffer<OriginValue> const origin_buffer, ::yampi::buffer<ResultValue> result_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      assert(origin_buffer.data() != result_buffer.data());

# if MPI_VERSION >= 4
      int const error_code
        = MPI_Rget_accumulate_c(
            origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
            result_buffer.data(), result_buffer.count().mpi_count(), result_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Rget_accumulate(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            result_buffer.data(), result_buffer.count(), result_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request::do_fetch_accumulate", environment);
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    static MPI_Request make_fetch_accumulate_request(
      ::yampi::buffer<OriginValue> const origin_buffer, ::yampi::buffer<ResultValue> result_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_fetch_accumulate(result, origin_buffer, result_buffer, target, target_buffer, operation, window, environment);
      return result;
    }

   public:
    using base_type::reset;

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_put_t const,
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      put(origin_buffer, target, target_buffer, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_get_t const,
      ::yampi::buffer<OriginValue> origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      get(origin_buffer, target, target_buffer, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_accumulate_t const,
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      accumulate(origin_buffer, target, target_buffer, operation, window, environment);
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_fetch_accumulate_t const,
      ::yampi::buffer<OriginValue> const origin_buffer, ::yampi::buffer<ResultValue> result_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      fetch_accumulate(origin_buffer, result_buffer, target, target_buffer, operation, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void put(
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    { do_put(mpi_request_, origin_buffer, target, target_buffer, window, environment); }

    template <typename OriginValue, typename TargetValue, typename Window>
    void get(
      ::yampi::buffer<OriginValue> origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    { do_get(mpi_request_, origin_buffer, target, target_buffer, window, environment); }

    template <typename OriginValue, typename TargetValue, typename Window>
    void accumulate(
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::ampi::rank const& target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    { do_accumulate(mpi_request_, origin_buffer, target, target_buffer, operation, window, environment); }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    void fetch_accumulate(
      ::yampi::buffer<OriginValue> const origin_buffer, ::yampi::buffer<ResultValue> result_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    { do_fetch_accumulate(mpi_request_, origin_buffer, result_buffer, target, target_buffer, operation, window, environment); }
  };

  inline void swap(::yampi::rma_request& lhs, ::yampi::rma_request& rhs) noexcept
  { lhs.swap(rhs); }

  class rma_request_ref
    : public ::yampi::request_ref_base
  {
    typedef ::yampi::request_ref_base base_type;

   public:
    rma_request_ref() = delete;
    ~rma_request_ref() noexcept = default;

    using base_type::base_type;
    using base_type::reset;

    void reset(::yampi::rma_request&& request, ::yampi::environment const& environment)
    {
      free(environment);
      *mpi_request_ptr_ = std::move(request.mpi_request_);
      request.mpi_request_ = MPI_REQUEST_NULL;
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_put_t const,
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      put(origin_buffer, target, target_buffer, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_get_t const,
      ::yampi::buffer<OriginValue> origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      get(origin_buffer, target, target_buffer, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_accumulate_t const,
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      accumulate(origin_buffer, target, target_buffer, operation, window, environment);
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    void reset(
      ::yampi::request_fetch_accumulate_t const,
      ::yampi::buffer<OriginValue> const origin_buffer, ::yampi::buffer<ResultValue> result_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      free(environment);
      fetch_accumulate(origin_buffer, result_buffer, target, target_buffer, operation, window, environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void put(
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Rput_c(
            origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), mpi_request_ptr_);
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Rput(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), mpi_request_ptr_);
# endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request_ref::put", environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void get(
      ::yampi::buffer<OriginValue> origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Rget_c(
            origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), mpi_request_ptr_);
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Rget(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            window.mpi_win(), mpi_request_ptr_);
# endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request_ref::get", environment);
    }

    template <typename OriginValue, typename TargetValue, typename Window>
    void accumulate(
      ::yampi::buffer<OriginValue> const origin_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
# if MPI_VERSION >= 4
      int const error_code
        = MPI_Raccumulate_c(
            origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), mpi_request_ptr_);
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Raccumulate(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), mpi_request_ptr_);
# endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request_ref::accumulate", environment);
    }

    template <typename OriginValue, typename ResultValue, typename TargetValue, typename Window>
    void fetch_accumulate(
      ::yampi::buffer<OriginValue> const origin_buffer, ::yampi::buffer<ResultValue> result_buffer,
      ::yampi::rank const target, ::yampi::target_buffer<TargetValue> const target_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::window_base<Window> const& window, ::yampi::environment const& environment)
    {
      assert(origin_buffer.data() != result_buffer.data());

# if MPI_VERSION >= 4
      int const error_code
        = MPI_Rget_accumulate_c(
            origin_buffer.data(), origin_buffer.count().mpi_count(), origin_buffer.datatype().mpi_datatype(),
            result_buffer.data(), result_buffer.count().mpi_count(), result_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count().mpi_count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), mpi_request_ptr_);
# else // MPI_VERSION >= 4
      int const error_code
        = MPI_Rget_accumulate(
            origin_buffer.data(), origin_buffer.count(), origin_buffer.datatype().mpi_datatype(),
            result_buffer.data(), result_buffer.count(), result_buffer.datatype().mpi_datatype(),
            target.mpi_rank(), target_buffer.displacement().mpi_displacement(), target_buffer.count(), target_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), window.mpi_win(), mpi_request_ptr_);
# endif // MPI_VERSION >= 4
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::rma_request_ref::fetch_accumulate", environment);
    }
  };

  inline void swap(::yampi::rma_request_ref& lhs, ::yampi::rma_request_ref& rhs) noexcept
  { lhs.swap(rhs); }

  class rma_request_cref
    : public ::yampi::request_cref_base
  {
    typedef ::yampi::request_cref_base base_type;

   public:
    rma_request_cref() = delete;
    ~rma_request_cref() noexcept = default;

    using base_type::base_type;
  };
}
# endif // MPI_VERSION >= 3


#endif

