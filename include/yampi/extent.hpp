#ifndef YAMPI_EXTENT_HPP
# define YAMPI_EXTENT_HPP

# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <mpi.h>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  class extent
  {
# if MPI_VERSION >= 3
    MPI_Count mpi_extent_;
# else
    MPI_Aint mpi_extent_;
# endif

   public:
    constexpr extent() : mpi_extent_{} { }

    extent(extent const&) = default;
    extent& operator=(extent const&) = default;
    extent(extent&&) = default;
    extent& operator=(extent&&) = default;
    ~extent() noexcept = default;

# if MPI_VERSION >= 3
    template <std::enable_if<not std::is_same<MPI_Aint, MPI_Count>::value, bool>::type = true>
    explicit constexpr extent(MPI_Aint const& mpi_extent) noexcept
      : mpi_extent_{static_cast<MPI_Count>(mpi_extent)}
    { }

    explicit constexpr extent(MPI_Count const& mpi_extent)
      noexcept(std::is_nothrow_copy_constructible<MPI_Count>::value)
      : mpi_extent_{mpi_extent}
    { }

    constexpr MPI_Count const& mpi_extent() const { return mpi_extent_; }
    constexpr MPI_Aint mpi_aint_mpi_extent() const { return static_cast<MPI_Aint>(mpi_extent_); }

    void swap(extent& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Count>::value)
    {
      using std::swap;
      swap(mpi_extent_, other.mpi_extent_);
    }
# else
    explicit constexpr extent(MPI_Aint const& mpi_extent)
      noexcept(std::is_nothrow_copy_constructible<MPI_Aint>::value)
      : mpi_extent_{mpi_extent}
    { }

    constexpr MPI_Aint const& mpi_extent() const { return mpi_extent_; }
    constexpr MPI_Aint mpi_aint_mpi_extent() const { return mpi_extent_; }

    void swap(extent& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_extent_, other.mpi_extent_);
    }
# endif

    constexpr bool operator==(extent const& other) const noexcept { return mpi_extent_ == other.mpi_extent_; }
    constexpr bool operator<(extent const& other) const noexcept { return mpi_extent_ < other.mpi_extent_; }

    extent& operator++() noexcept { ++mpi_extent_; return *this; }
    extent& operator--() noexcept { --mpi_extent_; return *this; }
    extent& operator+=(extent const& other) noexcept { mpi_extent_ += other.mpi_extent_; return *this; }
    extent& operator-=(extent const& other) noexcept { mpi_extent_ -= other.mpi_extent_; return *this; }

    template <typename Integer>
    extent& operator*=(Integer const scalar) noexcept { mpi_extent_ *= scalar; return *this; }

    template <typename Integer>
    extent& operator/=(Integer const scalar) noexcept { mpi_extent_ /= scalar; return *this; }

    template <typename Integer>
    extent& operator%=(Integer const scalar) noexcept { mpi_extent_ %= scalar; return *this; }
  };

  inline constexpr bool operator!=(::yampi::extent const& lhs, ::yampi::extent const& rhs) noexcept
  { return not (lhs == rhs); }

  inline constexpr bool operator>(::yampi::extent const& lhs, ::yampi::extent const& rhs) noexcept
  { return rhs < lhs; }

  inline constexpr bool operator<=(::yampi::extent const& lhs, ::yampi::extent const& rhs) noexcept
  { return not (lhs > rhs); }

  inline constexpr bool operator>=(::yampi::extent const& lhs, ::yampi::extent const& rhs) noexcept
  { return not (lhs < rhs); }

  inline ::yampi::extent operator++(::yampi::extent& self, int)
  { ::yampi::extent result = self; ++self; return result; }

  inline ::yampi::extent operator--(::yampi::extent& self, int)
  { ::yampi::extent result = self; --self; return result; }

  inline ::yampi::extent operator+(::yampi::extent lhs, ::yampi::extent const& rhs)
  { lhs += rhs; return lhs; }

  inline ::yampi::extent operator-(::yampi::extent lhs, ::yampi::extent const& rhs)
  { lhs -= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::extent operator*(::yampi::extent lhs, Integer const scalar)
  { lhs *= scalar; return lhs; }

  template <typename Integer>
  inline ::yampi::extent operator*(Integer const scalar, ::yampi::extent rhs)
  { rhs *= scalar; return rhs; }

  template <typename Integer>
  inline ::yampi::extent operator/(::yampi::extent lhs, Integer const scalar)
  { lhs /= scalar; return lhs; }

  template <typename Integer>
  inline ::yampi::extent operator%(::yampi::extent lhs, Integer const scalar)
  { lhs %= scalar; return lhs; }

  inline void swap(::yampi::extent& lhs, ::yampi::extent& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif

