#ifndef YAMPI_BYTE_DISPLACEMENT_HPP
# define YAMPI_BYTE_DISPLACEMENT_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/has_nothrow_copy.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <mpi.h>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
# else
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
# endif

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
    BOOST_CONSTEXPR byte_displacement() : mpi_byte_displacement_() { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    byte_displacement(byte_displacement const&) = default;
    byte_displacement& operator=(byte_displacement const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    byte_displacement(byte_displacement&&) = default;
    byte_displacement& operator=(byte_displacement&&) = default;
#   endif
    ~byte_displacement() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    BOOST_CONSTEXPR explicit byte_displacement(MPI_Aint const& mpi_byte_displacement)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Aint>::value)
      : mpi_byte_displacement_(mpi_byte_displacement)
    { }

    BOOST_CONSTEXPR bool operator==(byte_displacement const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_byte_displacement_ == other.mpi_byte_displacement_; }

    BOOST_CONSTEXPR bool operator<(byte_displacement const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_byte_displacement_ < other.mpi_byte_displacement_; }

    byte_displacement& operator++() BOOST_NOEXCEPT_OR_NOTHROW
    {
      mpi_byte_displacement_ += static_cast<MPI_Aint>(1);
      return *this;
    }

    byte_displacement& operator--() BOOST_NOEXCEPT_OR_NOTHROW
    {
      mpi_byte_displacement_ -= static_cast<MPI_Aint>(1);
      return *this;
    }

    byte_displacement& operator+=(byte_displacement const& other) BOOST_NOEXCEPT_OR_NOTHROW
    {
      mpi_byte_displacement_ += other.mpi_byte_displacement_;
      return *this;
    }

    byte_displacement& operator-=(byte_displacement const& other) BOOST_NOEXCEPT_OR_NOTHROW
    {
      mpi_byte_displacement_ -= other.mpi_byte_displacement_;
      return *this;
    }

    BOOST_CONSTEXPR MPI_Aint const& mpi_byte_displacement() const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_byte_displacement_; }

    void swap(byte_displacement& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_byte_displacement_, other.mpi_byte_displacement_);
    }
  };

  inline BOOST_CONSTEXPR bool operator!=(::yampi::byte_displacement const& lhs, ::yampi::byte_displacement const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  inline BOOST_CONSTEXPR bool operator>=(::yampi::byte_displacement const& lhs, ::yampi::byte_displacement const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs < rhs))
  { return not (lhs < rhs); }

  inline BOOST_CONSTEXPR bool operator>(::yampi::byte_displacement const& lhs, ::yampi::byte_displacement const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs < rhs))
  { return rhs < lhs; }

  inline BOOST_CONSTEXPR bool operator<=(::yampi::byte_displacement const& lhs, ::yampi::byte_displacement const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs < rhs))
  { return not (rhs < lhs); }

  inline ::yampi::byte_displacement operator++(::yampi::byte_displacement& self, int)
  { ::yampi::byte_displacement result = self; ++self; return result; }

  inline ::yampi::byte_displacement operator--(::yampi::byte_displacement& self, int)
  { ::yampi::byte_displacement result = self; --self; return result; }

  inline ::yampi::byte_displacement operator+(::yampi::byte_displacement lhs, ::yampi::byte_displacement const& rhs)
  { lhs += rhs; return lhs; }

  inline ::yampi::byte_displacement operator-(::yampi::byte_displacement lhs, ::yampi::byte_displacement const& rhs)
  { lhs -= rhs; return lhs; }

  inline void swap(::yampi::byte_displacement& lhs, ::yampi::byte_displacement& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_copy_constructible

#endif
