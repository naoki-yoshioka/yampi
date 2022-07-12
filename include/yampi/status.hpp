#ifndef YAMPI_STATUS_HPP
# define YAMPI_STATUS_HPP

# include <cstddef>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/count.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  class count_value_undefined_error
    : public std::runtime_error
  {
   public:
    count_value_undefined_error()
      : std::runtime_error{"count in MPI_GET_COUNT is MPI_UNDEFINED"}
    { }
  };

  class status
  {
    MPI_Status mpi_status_;

   public:
    status() = delete;

    explicit status(MPI_Status const& stat)
      noexcept(std::is_nothrow_copy_constructible<MPI_Status>::value)
      : mpi_status_{stat}
    { }

    status(status const&) = default;
    status& operator=(status const&) = default;
    status(status&&) = default;
    status& operator=(status&&) = default;
    ~status() noexcept = default;

    ::yampi::rank source() const
      noexcept(std::is_nothrow_copy_constructible< ::yampi::rank >::value)
    { return ::yampi::rank(mpi_status_.MPI_SOURCE); }

    ::yampi::tag tag() const
      noexcept(std::is_nothrow_copy_constructible< ::yampi::tag >::value)
    { return ::yampi::tag(mpi_status_.MPI_TAG); }

    void test_error(::yampi::environment const& environment) const
    {
      if (mpi_status_.MPI_ERROR != MPI_SUCCESS)
        throw ::yampi::error(mpi_status_.MPI_ERROR, "yampi::status::test_error", environment);
    }

    template <typename Datatype>
    ::yampi::count message_length(
      ::yampi::datatype_base<Datatype> const& datatype, ::yampi::environment const& environment) const
    {
      int result;
# if MPI_VERSION >= 3
      int const error_code
        = MPI_Get_count(std::addressof(mpi_status_), datatype.mpi_datatype(), &result);
# else // MPI_VERSION >= 3
      int const error_code
        = MPI_Get_count(const_cast<MPI_Status*>(std::addressof(mpi_status_)), datatype.mpi_datatype(), &result);
# endif // MPI_VERSION >= 3
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::status::message_length", environment);
      if (result == MPI_UNDEFINED)
        throw ::yampi::count_value_undefined_error();

      return ::yampi::count(result);
    }

    template <typename Datatype>
    ::yampi::count num_basic_elements(
      ::yampi::datatype_base<Datatype> const& datatype, ::yampi::environment const& environment) const
    {
# if MPI_VERSION >= 3
      MPI_Count result;
      int const error_code
        = MPI_Get_elements_x(std::addressof(mpi_status_), datatype.mpi_datatype(), &result);
# else
      int result;
      int const error_code
        = MPI_Get_elements(const_cast<MPI_Status*>(std::addressof(mpi_status_)), datatype.mpi_datatype(), &result);
# endif

      return error_code == MPI_SUCCESS
        ? ::yampi::count(result)
        : throw ::yampi::error(error_code, "yampi::status::num_basic_elements", environment);
    }

    bool empty() const
      noexcept(
        noexcept(mpi_status_.MPI_TAG == MPI_ANY_TAG)
        and noexcept(mpi_status_.MPI_SOURCE == MPI_ANY_SOURCE)
        and noexcept(mpi_status_.MPI_ERROR == MPI_SUCCESS))
    {
      return mpi_status_.MPI_TAG == MPI_ANY_TAG
        and mpi_status_.MPI_SOURCE == MPI_ANY_SOURCE
        and mpi_status_.MPI_ERROR == MPI_SUCCESS;
    }

    MPI_Status const& mpi_status() const { return mpi_status_; }

    void swap(status& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Status>::value)
    {
      using std::swap;
      swap(mpi_status_, other.mpi_status_);
    }
  };

  inline void swap(::yampi::status& lhs, ::yampi::status& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }


  struct ignore_status_t { };
# if __cplusplus >= 201703L
  inline constexpr ::yampi::ignore_status_t ignore_status{};
# else
  constexpr ::yampi::ignore_status_t ignore_status{};
# endif
}


# undef YAMPI_is_nothrow_swappable

#endif

