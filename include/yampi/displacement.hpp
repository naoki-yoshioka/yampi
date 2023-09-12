#ifndef YAMPI_DISPLACEMENT_HPP
# define YAMPI_DISPLACEMENT_HPP

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
  class displacement
  {
    MPI_Aint mpi_displacement_;

   public:
    constexpr displacement() : mpi_displacement_{} { }

    displacement(displacement const&) = default;
    displacement& operator=(displacement const&) = default;
    displacement(displacement&&) = default;
    displacement& operator=(displacement&&) = default;
    ~displacement() noexcept = default;

    explicit constexpr displacement(MPI_Aint const& mpi_displacement)
      noexcept(std::is_nothrow_copy_constructible<MPI_Aint>::value)
      : mpi_displacement_{mpi_displacement}
    { }

    explicit constexpr operator MPI_Aint() const noexcept { return mpi_displacement_; }

    constexpr bool operator==(displacement const& other) const noexcept
    { return mpi_displacement_ == other.mpi_displacement_; }

    constexpr bool operator<(displacement const& other) const noexcept
    { return mpi_displacement_ < other.mpi_displacement_; }

    displacement& operator++() noexcept
    {
      mpi_displacement_ += static_cast<MPI_Aint>(1);
      return *this;
    }

    displacement& operator--() noexcept
    {
      mpi_displacement_ -= static_cast<MPI_Aint>(1);
      return *this;
    }

    displacement& operator+=(displacement const& other) noexcept
    {
      mpi_displacement_ += other.mpi_displacement_;
      return *this;
    }

    displacement& operator-=(displacement const& other) noexcept
    {
      mpi_displacement_ -= other.mpi_displacement_;
      return *this;
    }

    template <typename Integer>
    displacement& operator*=(Integer const scalar) noexcept
    {
      mpi_displacement_ *= scalar;
      return *this;
    }

    template <typename Integer>
    displacement& operator/=(Integer const scalar) noexcept
    {
      mpi_displacement_ /= scalar;
      return *this;
    }

    template <typename Integer>
    displacement& operator%=(Integer const scalar) noexcept
    {
      mpi_displacement_ %= scalar;
      return *this;
    }

    constexpr MPI_Aint const& mpi_displacement() const noexcept
    { return mpi_displacement_; }

    void swap(displacement& other) noexcept(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_displacement_, other.mpi_displacement_);
    }
  };

  inline constexpr bool operator!=(::yampi::displacement const& lhs, ::yampi::displacement const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  inline constexpr bool operator>=(::yampi::displacement const& lhs, ::yampi::displacement const& rhs) noexcept(noexcept(lhs < rhs))
  { return not (lhs < rhs); }

  inline constexpr bool operator>(::yampi::displacement const& lhs, ::yampi::displacement const& rhs) noexcept(noexcept(lhs < rhs))
  { return rhs < lhs; }

  inline constexpr bool operator<=(::yampi::displacement const& lhs, ::yampi::displacement const& rhs) noexcept(noexcept(lhs < rhs))
  { return not (rhs < lhs); }

  inline ::yampi::displacement operator++(::yampi::displacement& self, int)
  { ::yampi::displacement result = self; ++self; return result; }

  inline ::yampi::displacement operator--(::yampi::displacement& self, int)
  { ::yampi::displacement result = self; --self; return result; }

  inline ::yampi::displacement operator+(::yampi::displacement lhs, ::yampi::displacement const& rhs)
  { lhs += rhs; return lhs; }

  inline ::yampi::displacement operator-(::yampi::displacement lhs, ::yampi::displacement const& rhs)
  { lhs -= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::displacement operator*(::yampi::displacement lhs, Integer const rhs)
  { lhs *= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::displacement operator/(::yampi::displacement lhs, Integer const rhs)
  { lhs /= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::displacement operator%(::yampi::displacement lhs, Integer const rhs)
  { lhs %= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::displacement operator*(Integer const lhs, ::yampi::displacement rhs)
  { rhs *= lhs; return rhs; }

  inline void swap(::yampi::displacement& lhs, ::yampi::displacement& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif
