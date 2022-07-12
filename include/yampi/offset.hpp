#ifndef YAMPI_OFFSET_HPP
# define YAMPI_OFFSET_HPP

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
  class offset
  {
    MPI_Offset mpi_offset_;

   public:
    constexpr offset() : mpi_offset_{} { }

    offset(offset const&) = default;
    offset& operator=(offset const&) = default;
    offset(offset&&) = default;
    offset& operator=(offset&&) = default;
    ~offset() noexcept = default;

    explicit constexpr offset(MPI_Offset const& mpi_offset)
      noexcept(std::is_nothrow_copy_constructible<MPI_Offset>::value)
      : mpi_offset_{mpi_offset}
    { }

    constexpr MPI_Offset const& mpi_offset() const { return mpi_offset_; }

    void swap(offset& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Offset>::value)
    {
      using std::swap;
      swap(mpi_offset_, other.mpi_offset_);
    }

    constexpr bool operator==(offset const& other) const noexcept { return mpi_offset_ == other.mpi_offset_; }
    constexpr bool operator<(offset const& other) const noexcept { return mpi_offset_ < other.mpi_offset_; }

    offset& operator++() noexcept { ++mpi_offset_; return *this; }
    offset& operator--() noexcept { --mpi_offset_; return *this; }

    offset& operator+=(offset const& other) noexcept { mpi_offset_ += other.mpi_offset_; return *this; }
    offset& operator-=(offset const& other) noexcept { mpi_offset_ -= other.mpi_offset_; return *this; }

    template <typename Integer>
    offset& operator*=(Integer const scalar) noexcept { mpi_offset_ *= scalar; return *this; }

    template <typename Integer>
    offset& operator/=(Integer const scalar) noexcept { mpi_offset_ /= scalar; return *this; }

    template <typename Integer>
    offset& operator%=(Integer const scalar) noexcept { mpi_offset_ %= scalar; return *this; }
  };

  inline constexpr bool operator!=(::yampi::offset const& lhs, ::yampi::offset const& rhs) noexcept
  { return not (lhs == rhs); }

  inline constexpr bool operator>(::yampi::offset const& lhs, ::yampi::offset const& rhs) noexcept
  { return rhs < lhs; }

  inline constexpr bool operator<=(::yampi::offset const& lhs, ::yampi::offset const& rhs) noexcept
  { return not (lhs > rhs); }

  inline constexpr bool operator>=(::yampi::offset const& lhs, ::yampi::offset const& rhs) noexcept
  { return not (lhs < rhs); }

  inline ::yampi::offset operator++(::yampi::offset& self, int)
  { ::yampi::offset result = self; ++self; return result; }

  inline ::yampi::offset operator--(::yampi::offset& self, int)
  { ::yampi::offset result = self; --self; return result; }

  inline ::yampi::offset operator+(::yampi::offset lhs, ::yampi::offset const& rhs)
  { lhs += rhs; return lhs; }

  inline ::yampi::offset operator-(::yampi::offset lhs, ::yampi::offset const& rhs)
  { lhs -= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::offset operator*(::yampi::offset lhs, Integer const scalar)
  { lhs *= scalar; return lhs; }

  template <typename Integer>
  inline ::yampi::offset operator*(Integer const scalar, ::yampi::offset rhs)
  { rhs *= scalar; return rhs; }

  template <typename Integer>
  inline ::yampi::offset operator/(::yampi::offset lhs, Integer const scalar)
  { lhs /= scalar; return lhs; }

  template <typename Integer>
  inline ::yampi::offset operator%(::yampi::offset lhs, Integer const scalar)
  { lhs %= scalar; return lhs; }

  inline void swap(::yampi::offset& lhs, ::yampi::offset& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif

