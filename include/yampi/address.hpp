#ifndef YAMPI_ADDRESS_HPP
# define YAMPI_ADDRESS_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/utility/is_nothrow_swappable.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  class address
  {
    MPI_Aint mpi_address_;

   public:
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    address() = default;
    address(address const&) = default;
    address& operator=(address const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    address(address&&) = default;
    address& operator=(address&&) = default;
#   endif
    ~address() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    explicit address(MPI_Aint const mpi_address) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_address_(mpi_address)
    { }

    BOOST_CONSTEXPR bool operator==(address const other) const
    { return mpi_address_ == other.mpi_address_; }

    BOOST_CONSTEXPR bool operator<(address const other) const
    { return mpi_address_ < other.mpi_address_; }

    address& operator++()
    {
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      mpi_address_ = MPI_Aint_add(mpi_address_, static_cast<MPI_Aint>(1));
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      MPI_Get_address(static_cast<char*>(mpi_address_) + static_cast<MPI_Aint>(1), &mpi_address_);
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      return *this;
    }

    address& operator--()
    {
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      mpi_address_ = MPI_Aint_diff(mpi_address_, static_cast<MPI_Aint>(1));
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      MPI_Get_address(
        static_cast<char*>(mpi_address_) - static_cast<char*>(static_cast<MPI_Aint>(1)),
        &mpi_address_);
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      return *this;
    }

    address& operator+=(address const other)
    {
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      mpi_address_ = MPI_Aint_add(mpi_address_, other.mpi_address_);
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      MPI_Get_address(static_cast<char*>(mpi_address_) + other.mpi_address_, &mpi_address_);
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      return *this;
    }

    address& operator-=(address const other)
    {
# if (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      mpi_address_ = MPI_Aint_diff(mpi_address_, other.mpi_address_);
# else // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      MPI_Get_address(
        static_cast<char*>(mpi_address_) - static_cast<char*>(other.mpi_address_),
        &mpi_address_);
# endif // (MPI_VERSION > 3) || (MPI_VERSION == 3 && MPI_SUBVERSION >= 1)
      return *this;
    }

    MPI_Aint const& mpi_address() const { return mpi_address_; }

    void swap(address& other)
      BOOST_NOEXCEPT_IF(::yampi::utility::is_nothrow_swappable<MPI_Aint>::value)
    {
      using std::swap;
      swap(mpi_address_, other.mpi_address_);
    }
  };

  inline BOOST_CONSTEXPR bool operator!=(::yampi::address const lhs, ::yampi::address const rhs)
  { return not (lhs == rhs); }

  inline BOOST_CONSTEXPR bool operator>=(::yampi::address const lhs, ::yampi::address const rhs)
  { return not (lhs < rhs); }

  inline BOOST_CONSTEXPR bool operator>(::yampi::address const lhs, ::yampi::address const rhs)
  { return rhs < lhs; }

  inline BOOST_CONSTEXPR bool operator<=(::yampi::address const lhs, ::yampi::address const rhs)
  { return not (rhs < lhs); }

  inline ::yampi::address operator++(::yampi::address& self, int)
  { ::yampi::address result = self; ++self; return result; }

  inline ::yampi::address operator--(::yampi::address& self, int)
  { ::yampi::address result = self; --self; return result; }

  inline ::yampi::address operator+(::yampi::address lhs, ::yampi::address const rhs)
  { lhs += rhs; return lhs; }

  inline ::yampi::address operator-(::yampi::address lhs, ::yampi::address const rhs)
  { lhs -= rhs; return lhs; }

  inline void swap(::yampi::address& lhs, ::yampi::address& rhs)
    BOOST_NOEXCEPT_IF(::yampi::utility::is_nothrow_swappable< ::yampi::address >::value)
  { lhs.swap(rhs); }
}


# undef YAMPI_addressof

#endif
