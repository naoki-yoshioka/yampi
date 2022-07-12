#ifndef YAMPI_BYTE_DISPLACEMENT_HPP
# define YAMPI_BYTE_DISPLACEMENT_HPP

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
  class byte_displacement
  {
    MPI_Aint mpi_byte_displacement_;

   public:
    constexpr byte_displacement() : mpi_byte_displacement_() { }

    byte_displacement(byte_displacement const&) = default;
    byte_displacement& operator=(byte_displacement const&) = default;
    byte_displacement(byte_displacement&&) = default;
    byte_displacement& operator=(byte_displacement&&) = default;
    ~byte_displacement() noexcept = default;

    explicit constexpr byte_displacement(MPI_Aint const& mpi_byte_displacement)
      noexcept(std::is_nothrow_copy_constructible<MPI_Aint>::value)
      : mpi_byte_displacement_(mpi_byte_displacement)
    { }

    explicit constexpr operator MPI_Aint() const noexcept { return mpi_byte_displacement_; }

    constexpr bool operator==(byte_displacement const& other) const noexcept
    { return mpi_byte_displacement_ == other.mpi_byte_displacement_; }

    constexpr bool operator<(byte_displacement const& other) const noexcept
    { return mpi_byte_displacement_ < other.mpi_byte_displacement_; }

    byte_displacement& operator++() noexcept
    {
      mpi_byte_displacement_ += static_cast<MPI_Aint>(1);
      return *this;
    }

    byte_displacement& operator--() noexcept
    {
      mpi_byte_displacement_ -= static_cast<MPI_Aint>(1);
      return *this;
    }

    byte_displacement& operator+=(byte_displacement const& other) noexcept
    {
      mpi_byte_displacement_ += other.mpi_byte_displacement_;
      return *this;
    }

    byte_displacement& operator-=(byte_displacement const& other) noexcept
    {
      mpi_byte_displacement_ -= other.mpi_byte_displacement_;
      return *this;
    }

    template <typename Integer>
    byte_displacement& operator*=(Integer const scalar) noexcept
    {
      mpi_byte_displacement_ *= scalar;
      return *this;
    }

    template <typename Integer>
    byte_displacement& operator/=(Integer const scalar) noexcept
    {
      mpi_byte_displacement_ /= scalar;
      return *this;
    }

    template <typename Integer>
    byte_displacement& operator%=(Integer const scalar) noexcept
    {
      mpi_byte_displacement_ %= scalar;
      return *this;
    }

    constexpr MPI_Aint const& mpi_byte_displacement() const noexcept
    { return mpi_byte_displacement_; }

    void swap(byte_displacement& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_byte_displacement_, other.mpi_byte_displacement_);
    }
  };

  inline constexpr bool operator!=(::yampi::byte_displacement const& lhs, ::yampi::byte_displacement const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  inline constexpr bool operator>=(::yampi::byte_displacement const& lhs, ::yampi::byte_displacement const& rhs) noexcept(noexcept(lhs < rhs))
  { return not (lhs < rhs); }

  inline constexpr bool operator>(::yampi::byte_displacement const& lhs, ::yampi::byte_displacement const& rhs) noexcept(noexcept(lhs < rhs))
  { return rhs < lhs; }

  inline constexpr bool operator<=(::yampi::byte_displacement const& lhs, ::yampi::byte_displacement const& rhs) noexcept(noexcept(lhs < rhs))
  { return not (rhs < lhs); }

  inline ::yampi::byte_displacement operator++(::yampi::byte_displacement& self, int)
  { ::yampi::byte_displacement result = self; ++self; return result; }

  inline ::yampi::byte_displacement operator--(::yampi::byte_displacement& self, int)
  { ::yampi::byte_displacement result = self; --self; return result; }

  inline ::yampi::byte_displacement operator+(::yampi::byte_displacement lhs, ::yampi::byte_displacement const& rhs)
  { lhs += rhs; return lhs; }

  inline ::yampi::byte_displacement operator-(::yampi::byte_displacement lhs, ::yampi::byte_displacement const& rhs)
  { lhs -= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::byte_displacement operator*(::yampi::byte_displacement lhs, Integer const rhs)
  { lhs *= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::byte_displacement operator/(::yampi::byte_displacement lhs, Integer const rhs)
  { lhs /= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::byte_displacement operator%(::yampi::byte_displacement lhs, Integer const rhs)
  { lhs %= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::byte_displacement operator*(Integer const lhs, ::yampi::byte_displacement rhs)
  { rhs *= lhs; return rhs; }

  inline void swap(::yampi::byte_displacement& lhs, ::yampi::byte_displacement& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif
