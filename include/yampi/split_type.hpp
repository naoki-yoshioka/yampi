#if MPI_VERSION >= 3
# ifndef YAMPI_SPLIT_TYPE_HPP
#   define YAMPI_SPLIT_TYPE_HPP

#   include <cassert>
#   include <string>
#   include <utility>
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
#   include <stdexcept>

#   include <mpi.h>

#   include <yampi/environment.hpp>

#   if __cplusplus >= 201703L
#     define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
#   else
#     define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
#   endif


namespace yampi
{
  struct shared_memory_split_type_t { };
  struct undefined_split_type_t { };

  namespace tags
  {
# if __cplusplus >= 201703L
    inline constexpr ::yampi::shared_memory_split_type_t shared_memory_split_type{};
    inline constexpr ::yampi::undefined_split_type_t undefined_split_type{};
# else
    constexpr ::yampi::shared_memory_split_type_t shared_memory_split_type{};
    constexpr ::yampi::undefined_split_type_t undefined_split_type{};
# endif
  }

  class split_type
  {
    int mpi_split_type_;

   public:
    constexpr split_type() noexcept : mpi_split_type_(0) { }

    explicit constexpr split_type(int const mpi_split_type) noexcept
      : mpi_split_type_(mpi_split_type)
    { }

    explicit constexpr split_type(::yampi::shared_memory_split_type_t const) noexcept
      : mpi_split_type_(MPI_COMM_TYPE_SHARED)
    { }

    explicit constexpr split_type(::yampi::undefined_split_type_t const) noexcept
      : mpi_split_type_(MPI_UNDEFINED)
    { }

    split_type(split_type const&) = default;
    split_type& operator=(split_type const&) = default;
    split_type(split_type&&) = default;
    split_type& operator=(split_type&&) = default;
    ~split_type() noexcept = default;

    constexpr bool operator==(split_type const& other) const noexcept
    { return mpi_split_type_ == other.mpi_split_type_; }

    constexpr int const& mpi_split_type() const noexcept { return mpi_split_type_; }

    void swap(split_type& other) noexcept(YAMPI_is_nothrow_swappable<int>::value)
    {
      using std::swap;
      swap(mpi_split_type_, other.mpi_split_type_);
    }
  };

  inline constexpr bool operator!=(::yampi::split_type const& lhs, ::yampi::split_type const& rhs) noexcept
  { return not (lhs == rhs); }

  inline void swap(::yampi::split_type& lhs, ::yampi::split_type& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }

# if __cplusplus >= 201703L
  inline constexpr ::yampi::split_type shared_memory_split_type{::yampi::tags::shared_memory_split_type};
  inline constexpr ::yampi::split_type undefined_split_type{::yampi::tags::undefined_split_type};
# else
  constexpr ::yampi::split_type shared_memory_split_type{::yampi::tags::shared_memory_split_type};
  constexpr ::yampi::split_type undefined_split_type{::yampi::tags::undefined_split_type};
# endif
}


#   undef YAMPI_is_nothrow_swappable

# endif
#endif

