#ifndef YAMPI_OFFSET_HPP
# define YAMPI_OFFSET_HPP

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
  class offset
  {
    MPI_Offset mpi_offset_;

   public:
    BOOST_CONSTEXPR offset() : mpi_offset_() { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    offset(offset const&) = default;
    offset& operator=(offset const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    offset(offset&&) = default;
    offset& operator=(offset&&) = default;
#   endif
    ~offset() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    explicit BOOST_CONSTEXPR offset(MPI_Offset const& mpi_offset)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Offset>::value)
      : mpi_offset_(mpi_offset)
    { }

    BOOST_CONSTEXPR MPI_Offset const& mpi_offset() const { return mpi_offset_; }

    void swap(offset& other) BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Offset>::value)
    {
      using std::swap;
      swap(mpi_offset_, other.mpi_offset_);
    }

    BOOST_CONSTEXPR bool operator==(offset const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_offset_ == other.mpi_offset_; }

    BOOST_CONSTEXPR bool operator<(offset const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_offset_ < other.mpi_offset_; }

    offset& operator++() BOOST_NOEXCEPT_OR_NOTHROW
    { ++mpi_offset_; return *this; }

    offset& operator--() BOOST_NOEXCEPT_OR_NOTHROW
    { --mpi_offset_; return *this; }

    offset& operator+=(offset const& other) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_offset_ += other.mpi_offset_; return *this; }

    offset& operator-=(offset const& other) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_offset_ -= other.mpi_offset_; return *this; }

    template <typename Integer>
    offset& operator*=(Integer const scalar) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_offset_ *= scalar; return *this; }

    template <typename Integer>
    offset& operator/=(Integer const scalar) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_offset_ /= scalar; return *this; }

    template <typename Integer>
    offset& operator%=(Integer const scalar) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_offset_ %= scalar; return *this; }
  };

  inline BOOST_CONSTEXPR bool operator!=(::yampi::offset const& lhs, ::yampi::offset const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs == rhs); }

  inline BOOST_CONSTEXPR bool operator>(::yampi::offset const& lhs, ::yampi::offset const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs < lhs; }

  inline BOOST_CONSTEXPR bool operator<=(::yampi::offset const& lhs, ::yampi::offset const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs > rhs); }

  inline BOOST_CONSTEXPR bool operator>=(::yampi::offset const& lhs, ::yampi::offset const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
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

  inline void swap(::yampi::offset& lhs, ::yampi::offset& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_copy_constructible

#endif

