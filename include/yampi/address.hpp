#ifndef YAMPI_ADDRESS_HPP
# define YAMPI_ADDRESS_HPP

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

# include <yampi/byte_displacement.hpp>

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
  class address
  {
    MPI_Aint mpi_address_;

   public:
    BOOST_CONSTEXPR address() : mpi_address_() { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    address(address const&) = default;
    address& operator=(address const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    address(address&&) = default;
    address& operator=(address&&) = default;
#   endif
    ~address() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    BOOST_CONSTEXPR explicit address(MPI_Aint const& mpi_address)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Aint>::value)
      : mpi_address_(mpi_address)
    { }

    BOOST_CONSTEXPR bool operator==(address const& other) const
      BOOST_NOEXCEPT_OR_NOTHROW/*BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_address_ == other.mpi_address_))*/
    { return mpi_address_ == other.mpi_address_; }

    BOOST_CONSTEXPR bool operator<(address const& other) const
      BOOST_NOEXCEPT_OR_NOTHROW/*BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_address_ < other.mpi_address_))*/
    { return mpi_address_ < other.mpi_address_; }

    // The following addition and difference member functions are not noexcept because MPI_Get_address might cause error (no exception though)
    address& operator++()
    {
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      mpi_address_ = MPI_Aint_add(mpi_address_, static_cast<MPI_Aint>(1));
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      mpi_address_ += static_cast<MPI_Aint>(1);
      //MPI_Get_address(static_cast<char*>(mpi_address_) + static_cast<MPI_Aint>(1), &mpi_address_);
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      return *this;
    }

    address& operator--()
    {
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      mpi_address_ = MPI_Aint_diff(mpi_address_, static_cast<MPI_Aint>(1));
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      mpi_address_ -= static_cast<MPI_Aint>(1);
      //MPI_Get_address(
      //  static_cast<char*>(mpi_address_) - static_cast<char*>(static_cast<MPI_Aint>(1)),
      //  &mpi_address_);
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      return *this;
    }

    address& operator+=(::yampi::byte_displacement const& displacement)
    {
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      mpi_address_ = MPI_Aint_add(mpi_address_, displacement.mpi_byte_displacement());
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      mpi_address_ += displacement.mpi_byte_displacement();
      //MPI_Get_address(static_cast<char*>(mpi_address_) + displacement.mpi_byte_displacement(), &mpi_address_);
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      return *this;
    }

    address& operator-=(::yampi::byte_displacement const& displacement)
    {
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      mpi_address_ = MPI_Aint_diff(mpi_address_, displacement.mpi_byte_displacement());
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      mpi_address_ -= displacement.mpi_byte_displacement();
      //MPI_Get_address(
      //  static_cast<char*>(mpi_address_) - static_cast<char*>(displacement.mpi_byte_displacement()),
      //  &mpi_address_);
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      return *this;
    }

    BOOST_CONSTEXPR MPI_Aint const& mpi_address() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_address_; }

    void swap(address& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_address_, other.mpi_address_);
    }
  };

  inline BOOST_CONSTEXPR bool operator!=(::yampi::address const& lhs, ::yampi::address const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  inline BOOST_CONSTEXPR bool operator>=(::yampi::address const& lhs, ::yampi::address const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs < rhs))
  { return not (lhs < rhs); }

  inline BOOST_CONSTEXPR bool operator>(::yampi::address const& lhs, ::yampi::address const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs < rhs))
  { return rhs < lhs; }

  inline BOOST_CONSTEXPR bool operator<=(::yampi::address const& lhs, ::yampi::address const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs < rhs))
  { return not (rhs < lhs); }

  inline ::yampi::address operator++(::yampi::address& self, int)
  { ::yampi::address result = self; ++self; return result; }

  inline ::yampi::address operator--(::yampi::address& self, int)
  { ::yampi::address result = self; --self; return result; }

  inline ::yampi::address operator+(::yampi::address address, ::yampi::byte_displacement const& displacement)
  { address += displacement; return address; }

  inline ::yampi::address operator+(::yampi::byte_displacement const& displacement, ::yampi::address const& address)
  { return address + displacement; }

  inline ::yampi::address operator-(::yampi::address address, ::yampi::byte_displacement const& displacement)
  { address -= displacement; return address; }

  inline ::yampi::byte_displacement operator-(::yampi::address const& lhs, ::yampi::address const& rhs)
  { return ::yampi::byte_displacement(lhs.mpi_address() - rhs.mpi_address()); }

  inline void swap(::yampi::address& lhs, ::yampi::address& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_copy_constructible

#endif
