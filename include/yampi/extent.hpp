#ifndef YAMPI_EXTENT_HPP
# define YAMPI_EXTENT_HPP

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
  class extent
  {
# if MPI_VERSION >= 3
    MPI_Count mpi_extent_;
# else
    MPI_Aint mpi_extent_;
# endif

   public:
    BOOST_CONSTEXPR extent() : mpi_extent_() { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    extent(extent const&) = default;
    extent& operator=(extent const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    extent(extent&&) = default;
    extent& operator=(extent&&) = default;
#   endif
    ~extent() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

# if MPI_VERSION >= 3
    explicit BOOST_CONSTEXPR extent(MPI_Aint const& mpi_extent) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_extent_(static_cast<MPI_Count>(mpi_extent))
    { }

    explicit BOOST_CONSTEXPR extent(MPI_Count const& mpi_extent)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Count>::value)
      : mpi_extent_(mpi_extent)
    { }

    BOOST_CONSTEXPR MPI_Count const& mpi_extent() const { return mpi_extent_; }
    BOOST_CONSTEXPR MPI_Aint mpi_aint_mpi_extent() const { return static_cast<MPI_Aint>(mpi_extent_); }

    void swap(extent& other) BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Count>::value)
    {
      using std::swap;
      swap(mpi_extent_, other.mpi_extent_);
    }
# else
    explicit BOOST_CONSTEXPR extent(MPI_Aint const& mpi_extent)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Aint>::value)
      : mpi_extent_(mpi_extent)
    { }

    BOOST_CONSTEXPR MPI_Aint const& mpi_extent() const { return mpi_extent_; }
    BOOST_CONSTEXPR MPI_Aint mpi_aint_mpi_extent() const { return mpi_extent_; }

    void swap(extent& other) BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_extent_, other.mpi_extent_);
    }
# endif

    BOOST_CONSTEXPR bool operator==(extent const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_extent_ == other.mpi_extent_; }

    BOOST_CONSTEXPR bool operator<(extent const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_extent_ < other.mpi_extent_; }

    extent& operator++() BOOST_NOEXCEPT_OR_NOTHROW
    { ++mpi_extent_; return *this; }

    extent& operator--() BOOST_NOEXCEPT_OR_NOTHROW
    { --mpi_extent_; return *this; }

    extent& operator+=(extent const& other) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_extent_ += other.mpi_extent_; return *this; }

    extent& operator-=(extent const& other) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_extent_ -= other.mpi_extent_; return *this; }

    template <typename Integer>
    extent& operator*=(Integer const scalar) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_extent_ *= scalar; return *this; }

    template <typename Integer>
    extent& operator/=(Integer const scalar) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_extent_ /= scalar; return *this; }

    template <typename Integer>
    extent& operator%=(Integer const scalar) BOOST_NOEXCEPT_OR_NOTHROW
    { mpi_extent_ %= scalar; return *this; }
  };

  inline BOOST_CONSTEXPR bool operator!=(::yampi::extent const& lhs, ::yampi::extent const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs == rhs); }

  inline BOOST_CONSTEXPR bool operator>(::yampi::extent const& lhs, ::yampi::extent const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs < lhs; }

  inline BOOST_CONSTEXPR bool operator<=(::yampi::extent const& lhs, ::yampi::extent const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs > rhs); }

  inline BOOST_CONSTEXPR bool operator>=(::yampi::extent const& lhs, ::yampi::extent const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
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

  inline void swap(::yampi::extent& lhs, ::yampi::extent& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


#endif

